   Copyright José A. R. Fonollosa <jarfo@yahoo.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

code version: 2.02
code date: 09-OCT-2013
installation instructions: python 2.7 code. No installation required
required python modules: numpy, pandas, sklearn, scipy
Tested on a Linux machine (Fedora 17) with python 2.7.3 and the following versions of the python libraries
numpy==1.6.2
pandas==0.10.0
scikit-learn==0.13.1
scipy==0.10.1

TRAINING (Aprox. 45 minutes)
- Download train, SUP1 and SUP2 data from Kaggle (cause-effect competition)
- Edit SETTINGS.json to indicate your data folders
- Train the models: python train.py train train1 train2

FAST TEST (first 9 entries of the validation data)
- python predict.py CEfinal_valid_pairs_head.csv CEfinal_valid_publicinfo_head.csv CEfinal_valid_predictions_head.csv
Time to process: 10 seconds
Results: (CEfinal_valid_predictions_head.csv rounded to 4 decimals places)
SampleID,Target
valid1, 0.70
valid2, 0.00
valid3, 0.00
valid4, 0.00
valid5, 0.00
valid6,-0.55
valid7,-0.14
valid8, 0.00
valid9,-0.01

See the data page of the Kaggle cause-effect competition for information about the data
