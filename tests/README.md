# Tests

This README explains how to run tests for `video_features`.

The motivation is, of course, to make sure that nothing breaks after we added some commits to the repo.
The test might not cover all possible use-cases yet all minimal working examples that we have in the docs
as well as google colab notebooks must be tested.

Now, the tests focus on comparing the shapes and the values of the output features (with some tolerance).
These are compared with the commited references in `./tests/*/reference/`.

We expect that the same code on different machines/setups might output different feature values
slightly exceeding the tolerance.
It is ok if you get the same values compared to the code state before your changes.
Also feel free to recalculate the references on your setup by removing the `./tests/*/reference/` folders and
toggling `TO_MAKE_REF` to `True` in each file.



# How to run tests?

You may use the conda environment that was installed locally or
the [Docker](https://v-iashin.github.io/video_features/meta/docker) container.

To test all models but PWC, run this:
```bash
# conda activate torch_zoo
pytest --ignore tests/pwc
```

To test PWC and I3D with the PWC flow:
```bash
# conda deactivate
# conda activate pwc
pytest tests/pwc tests/i3d
# conda deactivate
```

It may throw 1 warning for `torch_zoo` and 6 more warnings for `pwc`.

Also, remember that running the code with `show_pred` should yield something reasonable.

# How to make a test?

**New test for an old model**
Just add another row to the decorator `@pytest.mark.parametrize`.
Then, comment it and run the old tests and make sure they pass.
Next, uncomment the new line, toggle `TO_MAKE_REF` to `True`, remove the corresponding `reference` folder,
run the tests to make new references, and toggle `TO_MAKE_REF` back to `False`.
Note, the reference files might be >100MB in size making it hard to commit to github.


**New model**
Pick an implemented test from another model that is most similar to the new model and build on top of it.
Make sure to patch the config for the 'import API tests', otherwise you will get the same output
every time passing all tests.
Also, toggle `TO_MAKE_REF` to `True` and during the first run it will create reference values and
remove `reference` folder if it already exists because it will raise an error.
Note, the reference files might be >100MB in size making it hard to commit to github.
