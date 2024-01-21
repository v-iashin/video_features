# Tests

This README explains how to run tests for `video_features`.

The motivation is, of course, to make sure that nothing breaks after we added some commits to the repo.
The test might not cover all possible use-cases yet all minimal working examples that we have in the docs
as well as google colab notebooks must be tested.

Now, the tests focus on comparing the shapes and the values of the output features (with some tolerance).
We expect that the same code on different machines/setups might output different feature values
slightly exceeding the tolerance.
This makes comparing values a bit tricky.

For this reason, we use the following approach:
First, you should make reference values for the tests on the `master` branch, before you make changes.
Then, make changes you want (or checkout a branch) and run the tests which will compare the outputs of the
new code to the references from `master`.
These should pass if nothing is broken.

# Ok, how to run tests?

First, checkout to `master` and run tests with `TO_MAKE_REF` set to `True` in each file `tests/*/test_*.py`
for which you want to run a test (Find & Replace works for well me).
Then, checkout to the branch you want to test and run tests with `TO_MAKE_REF` set to `False`.

To run tests for a specific model (`vggish`), you may use the following command:
```bash
# conda activate torch_zoo
pytest tests/vggish
```

```bash
# conda activate video_features
pytest
```

Also, remember that running the code with `show_pred` should yield something reasonable.
You may use the conda environment that was installed locally or
the [Docker](https://v-iashin.github.io/video_features/meta/docker) container.

# How to make a test?

**New test for an old model**
Just add another row to the decorator `@pytest.mark.parametrize`.
Next, remove the corresponding `reference` folder, toggle `TO_MAKE_REF` to `True`,
run the tests to make new references, and toggle `TO_MAKE_REF` back to `False`.
Finally, rerun the tests to make sure that they pass.

**New model**
Pick an implemented test from another model that is most similar to the new model and build on top of it.
Also, toggle `TO_MAKE_REF` to `True`, remove the corresponding `reference` folder (if exists),
run the tests to make new references.
