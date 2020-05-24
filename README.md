# Final Assignment 
  
:arrow_right: Link to Custom Dataset: https://drive.google.com/drive/folders/1RbJHVxo91jhekv3_E9GZvujUDNDaxFQu

:arrow_right: Link to how the Dataset was created: https://github.com/Deeksha-Pandit/EVA4-S14/blob/master/Final/README.md 

:arrow_right: <h3>Data Formats:</h3>
All our images are in .jpg format
This was done to keep our dataset to a minimum size for further computations.

:arrow_right: <h3>Fetching the Data:</h3>
- So the biggest challenge of this assignment was how to load such huge amount of data ?
- First, I tried to load a small amount of data (say 20 images of each kind) into colab
- Then I tried to convert them into numpy arrays and tried to store them in a list - since we have to input 4 kind of images
- After storing in a list, when I tried to retrieve the images via these numpy arrays, I landed up in a lot of errors due to channel mis-match and resizing issues
- This created a lot of confusion and I thought I would give up on the whole assignment because I was stuck with errors for days
- Then I decided to finally change my approach and restarted the whole assignment from scratch
- This time I thought of loading the paths of the dataset into colab instead of saving them in a list and converting to arrays etc
- I stored the paths of the 4 kinds of images in variables and extracted each of these paths locally into colab
- This saves a lot of time and RAM , it just takes 3-4 mins to extract the zip files into colab without eating away any RAM space
- Such a relief, I could finally get hold of the dataset

:arrow_right: <h3>Custom Dataset class:</h3>
- I wrote a class named CustomDataset
- It contains __init__ method to initialise the transforms and also the 4 file paths which were created with the 4 types of files
- Also containts __len__ to store fg_bg length
- The __getitem__ function containts 'bg_index' which makes sure that each of the 100 background images is mapped to each of the fg-bg, fg-bg-mask and depth images
- This was done by using: bg_index = index//4000 (i.e for 1 bg images 4000 fg-bg (and mask and depth) would be mapped)
- Code snippet:
- Also if any transforms were available, the transforms would be applied to the images at this point in the code

:arrow_right: <h3>Albumentations:</h3>

:arrow_right: <h3>Train Test Split:</h3>
- So the requirement was to split the whole 400k dataset into 70:30 :: train:test
- As we had also done this for imagenet dataset in previous assignment, I implemented it in the same way
- First I split the len() of whole dataset into 70% and then the remaining 30% would be the test data
- Then after this I ran random_split() functionality on these 70:30 split images - to generate random images for test and train
- However, this was generating random images for every batch and so I had to use SEED: timetorch.manual_seed(0) so that a fixed set of images are generated per batch
- And then applied random_split() after SEED

:arrow_right: <h3>Dataloader:</h3>
- Then called the DataLoader item twice - once for train set and once for validation set
- First, I experimented for images of size (resized) 64x64 (to test with reduces size images) - for this I gave a larger batch_size of about 128
- This worked well , and was faster
- Then as for transfer learning, I did not resize the images but had to reduce batch_size greatly and made it 32 (so that cuda does not run out of memory)
- This completed the data loading process :)
