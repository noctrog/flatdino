"""Small helper to inspect Optax MultiSteps state structure.

Run with:
    uv run python -m flatdino.aux
to see how `gradient_step` and the micro-step counter (`mini_step`) behave in the optimizer state.
"""

import optax


def inspect_multisteps_state(every_k: int = 4):
    chain = optax.MultiSteps(optax.sgd(1.0), every_k_schedule=every_k)
    params = {"w": 1.0}
    state = chain.init(params)

    print("State type:", type(state))
    print("gradient_step:", optax.tree_utils.tree_get(state, "gradient_step"))
    print("mini_step:", optax.tree_utils.tree_get(state, "mini_step"))

    def apply_once(p, s):
        grads = {"w": 1.0}
        updates, s = chain.update(grads, s, p)
        new_params = optax.apply_updates(p, updates)
        return new_params, s

    params, state = apply_once(params, state)
    print("after 1 micro-step -> gradient_step:", optax.tree_utils.tree_get(state, "gradient_step"))
    print("after 1 micro-step -> mini_step:", optax.tree_utils.tree_get(state, "mini_step"))

    for _ in range(every_k - 1):
        params, state = apply_once(params, state)
    print(
        f"after {every_k} micro-steps -> gradient_step:",
        optax.tree_utils.tree_get(state, "gradient_step"),
    )
    print(
        f"after {every_k} micro-steps -> mini_step:",
        optax.tree_utils.tree_get(state, "mini_step"),
    )


if __name__ == "__main__":
    inspect_multisteps_state()
