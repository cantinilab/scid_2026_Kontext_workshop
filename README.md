# Spatial Cell ID 2026 — Kontext workshop

This folder contains everything you need for the Kontext hands-on session:
the slides, a dataset (8 sections of mouse heart, Xenium), the Kontext code, and the notebook.

**If you are running this from the VM, copy this directory in /mydatalocal and start from Step 4.** 

---

## What you need

- A laptop running **macOS, Linux or Windows**
- About **10 GB of free disk space**

---

## Step 1 — Open a terminal

Open a terminal.

- **macOS**: press `Cmd + Space`, type `Terminal`, press Enter.
- **Windows**: click Start, type `Miniforge Prompt` (it will exist after step 2 —
  for now use `Command Prompt`).
- **Linux**: press `Ctrl + Alt + T`.

---

## Step 2 — Install Miniforge (if you don't have Anaconda or Miniconda installed yet)

Go to <https://conda-forge.org/download/> and download the **Miniforge** installer
for your system. Run it and accept the default options.

When it finishes, **close the terminal and open a new one**. You should now see
`(base)` at the beginning of the line. Conda is ready.


---

## Step 3 — Go to the workshop folder

Download and unzip the `scid_2026_workshop` and open the directory in the terminal : 

```bash
cd path/scid_2026_workshop
```

Check you are in the right place:

```bash
ls
```

You should see data, notebooks, src and environment.yml listed.

---

## Step 4 — Create the environment

This single command downloads and installs Python and all the required packages
(scanpy, numpy, liana, …) into a private environment named kontext.

```bash
conda env create -f environment.yml
```

Now activate the environment:

```bash
conda activate kontext
```

The beginning of your line should change from (base) to (kontext).
You will need to run conda activate kontext every time you open a new terminal.


---

## Step 5 — Install Kontext and its notebook kernel

Make Kontext code importable from the notebook and the environment visible to Jupyter.

```bash 
pip install -e . --no-deps
python -m ipykernel install --user --name kontext --display-name "kontext"
```

## Step 6 — Launch the notebook

Click on Kontext jupyter 

or

Click on jupyter and choose Kontext on the top right button. 

or 

```bash 
jupyter notebook
```
