### cleanup todo

[x] put core training code into modular scripts
[ ] get training code working for SSN
[ ] get training code working for all other model types (MC-Drop, Evid, P-UNet) nice.
[ ] put evaluation code into modular scripts
[ ] replace eval logging with a dataframe that adds rows to a pd df.
[ ] put deterministic training framework into modular scripts scripts
[ ] put calibration model trainer scripts into modular scripts
[ ] add in the stochastic wrappers for: P-Unet, Ensemble, Evidential DL,  MC-Dropout
[ ] see how easy it is to add in ErfNet and Deeplabv3
[ ] pipeline for clinscores data
[ ] clean up the clinscores data notebook and move to analysis folder. Nice.
[ ] not everything has to be perfect, just get the repo in a form where I can do my work without everyhting needing copying all the time.


### models todo

TODO: put in the new model with the SSN wrapper, applying SSN to the last layer
allows building the base model separately from SSN itself. that way I can try ERFnet
and deeplabv3+
    
TODO: sort out putting the dataset setup into separate scripts

TODO: can I turn this into one single script where there is a model config that defines
base models and stochastic model wrappers and losses and just trains them all.

TODO: think about how to unify my models under one roof (and train deterministic models
to get the calibration results. Nice.

tidy this up and put all my todo comments in a big todo file somewhere in the readme along with
my notes from today from my notebooks. Nice.
go over and store links to those papers maria has sent me. Nice.

remember discussion about how initialization is important and not having the
    activation in the last layer can cause gradient problems etc etc look at the
    mean abs gradient in each layer while training I guess.
    
going off og what Miguel was talking about, can I do QC more intelligently with a linear model that uses the uncertianty map (even the clinical score features perhaps..?) to predict the dice, score, AVD score for the model's mean segmentation? NOICE NOICE NOICE.

think about how to encourage sample diversity with the loss function, what happens
for example if we mask out the output where p < 0.5, so that it only counts towards the output
for a sample if it is predicted highly??? Need to try to improve the sample diversity of ssn for example. Just try putting the SSN layer earlier on in the model to see how well it works.

I must adjust for domain because the domain is different for no stroke patients maybe?

(even if the umap explains the domain maybe cna say in general the up is a way to domain generalize??)

run umap on my clinscores data and see if there are any specific outliers... put it in 2d.

[ ] other architectures
[ ] augmentation as aleatoric
[ ] full bayesian neural network as epistemic
[ ] switch to a combo loss for all models e.g xent + dice loss / do deterministic model analysis for such as loss, compare it to the calibration curve!
[ ] or argue we can calibrate dice loss and others use it?
[ ] switch to using the umaps to 'calibrate' each stocahstic model instead of using samples which is invalid for structural data
[ ] use middle of model features to predict clin scores
[ ] do clin scores adjustment for each parameter (e.g age which it predicts well)
[ ] do transformer model to extract umap feature before linear model (or convolution with an argmax)
[ ] do quality control using clin scores / transformer or convolution with argmax
[ ] use clin-scores to improve calibration of individuals!
[ ] do inter-rater variability analysis!!!
[ ] think about 'analysis on external data' problem
[ ] will need to clean up training for each model in order for this to work!
[ ] will need to clean up evaluation for each model in order for this to work!
[ ] how robust are the uncertainty maps?? e.g, add noise to the model inputs, regen the maps and examine each metric that I compute!!! NOICE.
[ ] run the clinscores analysis on maria's new labelled data
[x] sort out my slurm cluster so that it doesn't copy files from other models and mess things up! (I can do all the rsyncing when the models are trained) (fixed with code to print best checkpoint, simplest solution!)
[ ] show a plot just in a local area to try to show more sample diversity (and do overlapping images like they do in more fancy papers etc etc nice
[ ] pca or umap on the clin scores table, does the umap highlight these individuals if any stand out?
