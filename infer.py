import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from accelerate import PartialState
from transformers import set_seed
from vertexregen.modeling import VertexRegenModel
from vertexregen.decode import (
    SequenceDecoder,
    get_prefix_allowed_tokens_fn,
    cleanup_mesh,
)

torch.autograd.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default="outputs/inference")
    parser.add_argument("-n", "--num-samples", type=int, default=32)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--no-guidance", action="store_true")
    args = parser.parse_args()

    state = PartialState()
    rank = state.process_index
    set_seed(args.seed + rank)

    output_path = Path(args.output)
    with state.local_main_process_first():
        output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VertexRegenModel.from_pretrained(args.model)
    model.set_attn_implementation("flash_attention_2")
    model.to(device)

    per_device_num_samples = (
        args.num_samples + state.num_processes - 1
    ) // state.num_processes

    decoder = SequenceDecoder(model.config, failfast=True)

    max_len = model.config.max_position_embeddings
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch_start in tqdm(
            range(0, per_device_num_samples, args.batch_size),
            position=state.process_index,
            desc="Generating samples",
        ):
            num_samples = min(args.batch_size, per_device_num_samples - batch_start)
            if args.no_guidance:
                constraint = None
            else:
                constraint = get_prefix_allowed_tokens_fn(
                    model.config, batch_size=num_samples
                )
            results = model.generate(
                max_new_tokens=max_len - 1,
                do_sample=True,
                top_p=0.95,
                top_k=100,
                num_return_sequences=num_samples,
                prefix_allowed_tokens_fn=constraint,
            )
            results = results.cpu()
            for i in range(num_samples):
                sample_id = batch_start + i
                decoded_results = decoder.run(results[i])
                num_vsplit = len(decoded_results) - 1
                init_vertices, init_faces = decoded_results[0]
                last_vertices, last_faces = decoded_results[-1]
                init_mesh = cleanup_mesh(init_vertices, init_faces, return_raw=True)
                init_mesh.export(
                    output_path
                    / f"sample_{rank:02d}_{sample_id:05d}_vsplit_step_init.ply"
                )
                final_mesh = cleanup_mesh(last_vertices, last_faces, return_raw=True)
                final_mesh.export(
                    output_path
                    / f"sample_{rank:02d}_{sample_id:05d}_vsplit_step_{num_vsplit:04d}.ply"
                )


if __name__ == "__main__":
    main()
