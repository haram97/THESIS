# Instructions
## Installation

### 1. Make a new python 3.8 environment:
if you use conda/anaconda:
> conda create -n myenv python=3.8

> conda activate myenv

else:
> python -m pip install --upgrade virtualenv

> virtualenv -p python3.8 myenv

> source myenv/venv/bin/activate 

or if you use Windows OS, replace last command with the following one
> myenv\Scripts\activate.bat

        
### 2. Install the required libraries using the following commands:
Make sure you navigate to the project directory using *cd*
> pip install -r requirements.txt

> python -m spacy download en_core_web_sm

### 3. Prepare your document
Prepare the input document into linebreak-separated requirements (*context*) in a 'txt' file format. We recommand to place it in the project directory.

Note: avoid long using context as the resolution model doesn't support more than input 512 tokens.

### 4. Run the artifact:
> python taphsir.py --doc path-to-doc --mode mode --detection dfeatures

You can select three options for **mode**:

To run detection pipeline only, select **1**. For resolution only, select **2**. And to run both, select **3**. Default value is **3**.

You can also select the detection features (dfeatures) if the selected mode includes detection. **LF** to run the detection using language features only. **FE** to use features embedding only. **Ensemble** to use both (default value).

Here are some usage examples:
> python taphsir.py --doc Example.txt --mode 2

This simply runs the resolution pipeline.
> python taphsir.py --doc Example.txt --mode 1 --detection LF   

This runs detection only using language features.
> python taphsir.py --doc /path/to/file.txt 

This runs both detection and resolution and detection method = ensemble.

### 5. The results are stored in Excel format in the output folder 
Both detection and resolution output Excel files contain the following columns:

**Id** is a unique Id with three components formatted as follows: 

*"context number"-"the pronoun"-"pronoun number within the context"*

E.g., *9-it-2* refers to the second pronoun *"it"* within the ninth Context.

**Context** refers to the paragraphs (sentences split by linebreaks) defined in the input file.

**Pronoun** is the analysed pronoun.

The detection file has **Result** column, it has two possible values **Ambiguous** or **Unambiguous**.

While the resolution file has **Predicted** column. This column shows the *select span* (phrase) as *resolution* (antecedent to the pronoun).

## Docker
If you are familiar with Docker containers, you can use [TAPHSIR image from the docker hub.](https://hub.docker.com/r/ezzini/taphsir)

Here are some basic usage commands:

> docker pull ezzini/taphsir

> docker run -d ezzini/taphsir

> docker start "$(docker ps -l -q)"

> docker cp "/path/to/file.txt" ezzini/taphsir:file.txt

> docker exec "$(docker ps -l -q)" python --doc file.txt --mode <mode> --detection <dfeatures>

> docker cp ezzini/taphsir:output "/desktop/taphsir/output"



