# PyTorch-Exercise
Collections of PyTorch exercises for personal learning mainly in M1 Max Mac Studio with OS X arm64 (Apple Silicon) for MUSIQ(Multi-scale Image Quality Transformer).

```
python3 -m venv venv && source venv/bin/activate && python3 -m pip install -r requirements_convert.txt

python3 convert_teacher_model.py
```

```
pip uninstall -y coremltools && pip install coremltools==7.1
```

It seems there's an issue with Python 3.13 and coremltools. Let's try creating a new virtual environment with Python 3.9, which is known to work well with coremltools:
```
python3.9 -m venv venv_py39 && source venv_py39/bin/activate && pip install -r requirements_convert.txt
```


```
(venv_py39) ➜  TeacherModel git:(main) ✗ python3 convert_teacher_model.py
Torch version 2.8.0 has not been tested with coremltools. You may run into unexpected errors. Torch 2.5.0 is the most recent version that has been tested.
```


Error converting to Core ML: For an ML Program, extension must be .mlpackage (not .mlmodel). Please see https://coremltools.readme.io/docs/unified-conversion-api#target-conversion-formats to see the difference between neuralnetwork and mlprogram model types.
```
        # Convert the model
        coreml_model = ct.convert(
            traced_model,
            inputs=[image_input],
            compute_units=ct.ComputeUnit.ALL,  # Use all available compute units
            minimum_deployment_target=ct.target.iOS16,  # Set minimum iOS version
            convert_to="mlprogram"  # Explicitly set to mlprogram format
        )
```
