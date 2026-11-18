# CSC 451 Docker test app

Use this app to audit the local candidate collection, test image uploads, and run a trained five-class ResNet-18 on CPU. The app runs on your computer at [localhost:8501](http://localhost:8501).

## Build and open

Run these commands from the project folder containing `app/`, with Docker Desktop running.

```powershell
docker build -t csc451-test ./app
docker run --rm -p 127.0.0.1:8501:8501 csc451-test
```

**Image test** accepts one dermoscopic JPEG or PNG. A mounted checkpoint enables top-three model scores. With no checkpoint, this view checks and previews the file.

**Dataset audit** counts candidate classes, review status, missing files, and patient linkage in a mounted collection. The audit leaves the CSV and images as they are.

## Mount the collection privately

Set the path to the private folder containing `manifest.csv` and `images/`. Replace the example path with the location of your private copy. The [data guide](../data/README.md) shows how to restore it from the team manifest.

```powershell
$dataDir = (Resolve-Path 'C:\path\to\private\isic-dermoscopy-4000').Path
docker run --rm -p 127.0.0.1:8501:8501 --mount "type=bind,source=$dataDir,target=/data,readonly" csc451-test
```

Keep the candidate collection in private storage while the course's source and consent requirements are being resolved. The image build context is `app/` and contains application code, configuration, and tests.

## Train and load a checkpoint

Once the instructor accepts the source and the team reviews the data, create `reviewed.csv` in the dataset folder with these columns:

```csv
relative_path,proposed_class,group_id,split
images/EXAMPLE.jpg,Melanoma,example-patient-group,train
```

Use the exact five project class names from the root README, paths relative to the CSV, and a resolved patient group for each included image. Use `train` and `val` groups with all five classes represented in each. An optional `test` split is checked for group overlap and reserved for final evaluation. The trainer rejects duplicate paths and group IDs shared across splits. Resolve records without patient linkage before including them in a patient-level run.

Train from the project folder using a writable checkpoint directory. Stop the earlier app container before starting another on the same port.

```powershell
$modelsDir = (New-Item -ItemType Directory -Force 'C:\path\to\private\csc451-models').FullName
docker run --rm --mount "type=bind,source=$dataDir,target=/data,readonly" --mount "type=bind,source=$modelsDir,target=/models" csc451-test python train.py --manifest /data/reviewed.csv --output /models/resnet18.pt --history /models/run.json --epochs 1 --baselines
docker run --rm -p 127.0.0.1:8501:8501 --mount "type=bind,source=$dataDir,target=/data,readonly" --mount "type=bind,source=$modelsDir,target=/models,readonly" csc451-test
```

`--baselines` evaluates the majority-class and logistic-regression baselines on validation data. The history JSON records per-epoch training/validation losses and validation metrics. Add `--pretrained` for the frozen ImageNet backbone planned in the proposal. The first pretrained run downloads the initialization weights. Longer experiments can use Colab and copy the resulting checkpoint into the private models folder.

## Reproduce the engineering pilot

The [pilot report](../docs/pilot-results.md) records the actual CPU run. The preparation script selects the same patients in the same order from the original private manifest. Use a new directory outside the checkout. This is the engineering split while team review and instructor approval are pending.

```powershell
$pilotDir = 'C:\path\to\private\csc451-pilot'
python data/prepare_pilot.py --source $dataDir --output $pilotDir
$modelsDir = Join-Path $pilotDir 'models'
docker run --rm -e TORCH_HOME=/models/torch-cache --mount "type=bind,source=$pilotDir,target=/data,readonly" --mount "type=bind,source=$modelsDir,target=/models" csc451-test python train.py --manifest /data/pilot-split.csv --output /models/resnet18.pt --history /models/run.json --epochs 3 --batch-size 16 --seed 451 --device cpu --pretrained --fine-tune-layer4 --baselines
```

The training command selects a checkpoint using validation macro-F1 and melanoma recall. Evaluate the saved checkpoint on the held-out split once.

```powershell
docker run --rm --network none --mount "type=bind,source=$pilotDir,target=/data,readonly" --mount "type=bind,source=$modelsDir,target=/models" csc451-test python evaluate.py --checkpoint /models/resnet18.pt --manifest /data/pilot-split.csv --output /models/holdout.json --initial-weights /models/torch-cache/hub/checkpoints/resnet18-f37072fd.pth
```

Start the app with the read-only dataset and model mounts above. Use the chosen private models folder or ask a teammate for the existing `resnet18.pt` pilot checkpoint.

## Run tests

```powershell
docker run --rm --network none csc451-test python -m unittest discover -s tests -v
python -m unittest discover -s data -p test_restore_collection.py -v
```
