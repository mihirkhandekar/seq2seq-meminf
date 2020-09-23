# Membership Inference Attacks on Seq-to-Seq Models

## Installation and Requirements
* System Requirements : TensorFlow 2.1+, Python 3.5+
Installation 
1. Create and activate a Virtual environment
> python3 -m virtualenv venv
> source venv/bin/activate
2. Install dependencies
> pip install -r requirements.txt

## Running the attack
### To train machine translation target model
`python main.py`

### To test attack 1 (Using average rank thresholding)
`python attack1.py`

## To test attack 2 (Shadow models - rank histogram)
`python attack2.py`

## Using the tool on an existing model
In progress

## File Structure
In progress

## TODOs 
* Restructure tool
* Check if attacks are correctly implemented
* Make usage of other format datasets easier
* Comments, documentation
* etc.

## Misc

Files adapted from https://github.com/csong27/auditing-text-generation:

- `helper.py`: defines custom architecture for user level NMT
- `load_sated.py1`: dataloading from SATED dataset
- `sated_nmt.py`: trains target and shadow models for SATED dataset
- `sated_nmt_ranks.py`: gets ranks for target and shadow models
