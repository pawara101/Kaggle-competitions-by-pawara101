FORCED_FUSED_RANK = 32


def _compress_lora_pair_to_rank(B: torch.Tensor, A_mat: torch.Tensor, rank: int):
    # Compute delta = B @ A
    delta = B.float() @ A_mat.float()

    # Perform SVD decomposition
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

    # Truncate to desired rank
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    # Reconstruct LoRA matrices
    sroot = torch.sqrt(S)
    B_new = U * sroot.unsqueeze(0)
    A_new = sroot.unsqueeze(1) * Vh

    return B_new.to(B.dtype).contiguous(), A_new.to(A_mat.dtype).contiguous()



def patched_merge_fused_projections(
    fused_model_key: str,
    adapter_layer_prefix: str,
    components,
    model_state_shapes,
    peft_weights,
    target_modules,
    profile,
) -> int:

    fused_out_dim = model_state_shapes[fused_model_key][0]
    fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]

    # Find component order
    component_order = None
    for target, comps in profile.fused_projection_map:
        if target == fused_target_name:
            component_order = comps
            break

    assert component_order is not None, "Component order not found"

    comp_by_name = {name: (lora_A, lora_B) for name, lora_A, lora_B in components}

    lora_A_parts = []
    comp_slices = []
    merged_rank = 0
    row_offset = 0

    # Merge components
    for comp_name in component_order:
        if comp_name not in comp_by_name:
            raise RuntimeError(f"Missing component {comp_name}")

        lora_A, lora_B = comp_by_name[comp_name]

        r = lora_A.shape[0]
        out_dim = lora_B.shape[0]

        lora_A_parts.append(lora_A)
        comp_slices.append((row_offset, row_offset + out_dim, r))

        row_offset += out_dim
        merged_rank += r

    merged_lora_A = torch.cat(lora_A_parts, dim=0)

    merged_lora_B = torch.zeros(
        fused_out_dim,
        merged_rank,
        dtype=merged_lora_A.dtype,
        device=merged_lora_A.device,
    )

    # Fill B matrix
    rank_offset = 0
    for i, (row_start, row_end, r) in enumerate(comp_slices):
        _, lora_B = comp_by_name[component_order[i]]
        merged_lora_B[row_start:row_end, rank_offset:rank_offset + r] = lora_B
        rank_offset += r

    # Compress to rank 32 if needed
    final_rank = merged_rank
    if merged_rank > FORCED_FUSED_RANK:
        merged_lora_B, merged_lora_A = _compress_lora_pair_to_rank(
            merged_lora_B,
            merged_lora_A,
            FORCED_FUSED_RANK
        )
        final_rank = FORCED_FUSED_RANK

    # Save result
    peft_target_key = f"{adapter_layer_prefix}.{fused_target_name}.weight"
    A._add_peft_weight(
        peft_target_key,
        merged_lora_A,
        merged_lora_B,
        peft_weights,
        target_modules
    )

    return final_rank


A._merge_fused_projections = patched_merge_fused_projections
print("Patched function:", A._merge_fused_projections.__name__)


weights.build_lora_adapter(
    base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    adapter_path="/kaggle/input/models/huikang/nemotron-adapter/transformers/default/27",
    output_path="/kaggle/working/nemotron-adapter-ready-to-submit",
)



shutil.make_archive(
    '/kaggle/working/submission',
    'zip',
    '/kaggle/working/nemotron-adapter-ready-to-submit'
)

print("ZIP archive successfully created!")