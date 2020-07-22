def form_slices(size: int, stack_size: int, step_size: int) -> list((int, int)):
    slices = []
    # calc how many full stacks can be formed out of framepaths
    full_stack_num = (size - stack_size) // step_size + 1
    for i in range(full_stack_num):
        start_idx = i * step_size
        end_idx = start_idx + stack_size
        slices.append((start_idx, end_idx))
    return slices


if __name__ == "__main__":
    print(form_slices(100, 15, 15))
