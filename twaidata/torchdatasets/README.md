the whole brain dataset loader returns a whole 3D brain scan.
I also need to do 2D slice dataset and add an option to take particular chunks of the brain
for the 3D version. To do this I need to pay careful attention to exactly which part of the brain I use
and also need to think about trying a 2.5D dataset at some point which will probably need some reshaping but
that is for another time....

note that i am use nibabel not sitk as in the group, and it might be possible to get it to load the image
as a float32 directrly i haven't checked, and I haveen't explore the performance issues.

I need to be aware that currently the output shapes from the dataset won't be in the format torch expects and that
might be a reason to use sitk as I think it uses a different ordering than nibabal which aligns with pytorch's
so that might work better.

however this is rough work, most of the logic is there, just the shaping bit to sort and testing. nice.

note that the datset expects a structure where the images are in a "imgs" folder within the datset root
and the labels are in an "labels" folder.
currently only one label file is allowed, this could be easily modified.
the files that the dataloader loads must be in the form `"{ind}_{filetype}"` where ind (individual) is say `CVD001` and filetype is
say `FLAIR.nii.gz`. so don't use underscores in individual names.