# Import required libraries
import pandas as pd
import numpy as np
import os
import shutil
import librosa
# Set number of columns to show in the notebook
pd.set_option('display.max_columns', 200)
# Set number of rows to show in the notebook
pd.set_option('display.max_rows', 50) 
# Make the graphs a bit prettier
# pd.set_option('display.mpl_style', 'default') 

# Import MatPlotLib Package
import matplotlib.pyplot as plt

# Display pictures within the notebook itself
get_ipython().run_line_magic("matplotlib", " inline")


# Read the annotations file
newdata = pd.read_csv('clip_info_final.csv', sep="\t")


# Display the top 5 rows
newdata.head(5)


# Get to know the data better
newdata.info()


# What colums are there ?
newdata.columns


# Extract the clip_id and mp3_path
newdata[["clip_id", "mp3_path"]]


# Previous command extracted it as a Dataframe. We need it as a matrix to do analyics on. 
# Extract clip_id and mp3_path as a matrix.
clip_id = newdata["clip_id"].values
mp3_path = newdata["mp3_path"].values


# Some of the tags in the dataset are really close to each other. Lets merge them together
synonyms = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['india', 'indian'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singer', 'singing'],
            ['space', 'spacey'],
            ['string', 'strings'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['strange', 'weird']]


# Merge the synonyms and drop all other columns than the first one.
"""
Example:
Merge 'beat', 'beats' and save it to 'beat'.
Merge 'classical', 'clasical', 'classic' and save it to 'classical'.
"""
for s in synonyms:
    newdata[s[0]] = newdata[s].max(axis=1)
    newdata.drop(s[1:], axis=1, inplace=True)


# Did it work ?
newdata.info()


# Lets view it.
newdata.head()


# Drop the mp3_path tag from the dataframe
newdata.drop('mp3_path', axis=1, inplace=True)
# Save the column names into a variable
data = newdata.sum(axis=0)


# Find the distribution of tags.
data


# Sort the column names.
data.sort_values(axis=0, inplace=True)


# Find the top tags from the dataframe.
topindex, topvalues = list(data.index[84:]), data.values[84:]
del(topindex[-1])
topvalues = np.delete(topvalues, -1)


# Get the top column names
topindex


# Get only the top column values
topvalues


# Get a list of columns to remove
rem_cols = data.index[:84]


# Cross-check: How many columns are we removing ?
len(rem_cols)


# Drop the columns that needs to be removed
newdata.drop(rem_cols, axis=1, inplace=True)


newdata.info()


# Create a backup of the dataframe
backup_newdata = newdata


# Shuffle the dataframe
from sklearn.utils import shuffle
newdata = shuffle(newdata)


newdata.reset_index(drop=True)


# One final check
newdata.info()


# Let us save the final columns
final_columns_names = list(newdata.columns)


# Do it only once to delete the clip_id column
del(final_columns_names[0])


# Verified
final_columns_names


# Create the file which is to be saved off (you could skip and apply similar steps in the previous dataframe)
# Here, binary 0's and 1's from each column is changed to 'False' and 'True' by using '==' operator on the dataframe.
final_matrix = pd.concat([newdata['clip_id'], newdata[final_columns_names]==1], axis=1)


# Rename all the mp3 files to their clip_id and save it into one folder named 'dataset_clip_id_mp3' in the same folder.
# Get the current working directory
root = os.getcwd()
os.mkdir( root + "/dataset_clip_id_mp3/", 0755 )

# Iterate over the mp3 files, rename them to the clip_id and save it to another folder.
for id in range(25863):
    #print clip_id[id], mp3_path[id]
    src = root + "/" + mp3_path[id]
    dest = root + "/dataset_clip_id_mp3/" + str(clip_id[id]) + ".mp3"
    shutil.copy2(src,dest)
    #print src,dest


# Convert all the mp3 files into their corresponding mel-spectrograms (melgrams).

# Audio preprocessing function
def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


# Get the absolute path of all audio files and save it to audio_paths array
audio_paths = []
# Variable to save the mp3 files that don't work
files_that_dont_work=[]
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
root = os.getcwd()
os.chdir(root + '/dataset_clip_id_mp3/')
for audio_path in os.listdir('.'):
    #audio_paths.append(os.path.abspath(fname))
    if os.path.isfile(root + '/dataset_clip_id_melgram/' + str(os.path.splitext(audio_path)[0]) + '.npy'):
        #print "existtt"
        continue
    else:
        if str(os.path.splitext(audio_path)[1]) == ".mp3":
            try:
                melgram = compute_melgram(os.path.abspath(audio_path))
                dest = root + '/dataset_clip_id_melgram/' + str(os.path.splitext(audio_path)[0])
                np.save(dest, melgram)
            except EOFError:
                files_that_dont_work.append(audio_path)
                continue
                
"""
NOTE: I've run this an created all the mel-spectrograms and saved them off seprately, 
and then concatenated the train, test and validation set in the ratio that I wanted.
This, will make a significant overhead in the computation time when you look at this
as a whole. 

For example, concatenating the corresponding files to train, test and
validation splits will inturn require more time and memory. If we decide the splits 
beforehand and converting mp3 to mel-spectrogram based on those splits, it will make
life much easier (and less time). 

However, I want each of the mel-spectrograms seperate as I might need to create datasets
based on different genre, number of files, splits etc. in the future. So this is the way
to go for me now. Please note that this requires a significant amount of system memory.
"""


# Get a list of 
mp3_available = []
melgram_available = []
for mp3 in os.listdir('/home/cc/notebooks/MusicProject/MagnaTagATune/dataset_clip_id_mp3/'):
     mp3_available.append(int(os.path.splitext(mp3)[0]))
        
for melgram in os.listdir('/home/cc/notebooks/MusicProject/MagnaTagATune/dataset_clip_id_melgram/'):
     melgram_available.append(int(os.path.splitext(melgram)[0]))


# The latest clip_id
new_clip_id = final_matrix['clip_id']


# Let us see which all files have not been converted into melspectrograms.
set(list(new_clip_id)).difference(melgram_available)


# Saw that these clips were extra 35644, 55753, 57881. Removing them.
final_matrix = final_matrix[final_matrix['clip_id']get_ipython().getoutput("= 35644]")
final_matrix = final_matrix[final_matrix['clip_id']get_ipython().getoutput("= 55753]")
final_matrix = final_matrix[final_matrix['clip_id']get_ipython().getoutput("= 57881]")


# Check again
final_matrix.info()


# Save the matrix
final_matrix.to_pickle('final_Dataframe.pkl')


# Seperate the training, test and validation dataframe.
training_with_clip = final_matrix[:19773]


validation_with_clip = final_matrix[19773:21294]


testing_with_clip = final_matrix[21294:]


# Quick peek
training_with_clip


# Quick peek
testing_with_clip


# Quick peek
validation_with_clip


# Extract the corresponding clip_id's
training_clip_id = training_with_clip['clip_id'].values
validation_clip_id = validation_with_clip['clip_id'].values
testing_clip_id = testing_with_clip['clip_id'].values


# Check !
training_clip_id


# Go to the directory you want to save the dataframe
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/final_dataset/')


# Save the 'y' values.
np.save('train_y.npy', training_with_clip[final_columns_names].values)


np.save('valid_y.npy', validation_with_clip[final_columns_names].values)


np.save('test_y.npy', testing_with_clip[final_columns_names].values)


# Save the 'x' clip_id's. We will make the numpy array using this.
np.savetxt('train_x_clip_id.txt', training_with_clip['clip_id'].values, fmt='get_ipython().run_line_magic("i')", "")


np.savetxt('test_x_clip_id.txt', testing_with_clip['clip_id'].values, fmt='get_ipython().run_line_magic("i')", "")


np.savetxt('valid_x_clip_id.txt', validation_with_clip['clip_id'].values, fmt='get_ipython().run_line_magic("i')", "")


# Now to combine the melgrams according to the clip_id. 
# (maybe in the future we can make melgrams according to the clip id iteslf into train test and validationget_ipython().getoutput("!)")

# Variable to store melgrams.
train_x = np.zeros((0, 1, 96, 1366))
test_x = np.zeros((0, 1, 96, 1366))
valid_x = np.zeros((0, 1, 96, 1366))

root = '/home/cc/notebooks/MusicProject/MagnaTagATune/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,valid_clip in enumerate(list(validation_clip_id)):
    if os.path.isfile(str(valid_clip) + '.npy'):
        #print i,valid_clip
        melgram = np.load(str(valid_clip) + '.npy')
        valid_x = np.concatenate((valid_x, melgram), axis=0)
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
np.save('valid_x.npy', valid_x)
print "Validation file created"


root = '/home/cc/notebooks/MusicProject/MagnaTagATune/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,test_clip in enumerate(list(testing_clip_id)):
    if os.path.isfile(str(test_clip) + '.npy'):
        #print i,test_clip
        melgram = np.load(str(test_clip) + '.npy')
        test_x = np.concatenate((test_x, melgram), axis=0)
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
np.save('test_x.npy', test_x)
print "Testing file created"

root = '/home/cc/notebooks/MusicProject/MagnaTagATune/'
os.chdir(root + "/dataset_clip_id_melgram/")
for i,train_clip in enumerate(list(training_clip_id)):
    #if os.path.isfile(str(train_clip) + '.npy'):
        #print i,train_clip
    melgram = compute_melgram(str(train_clip) + '.mp3')
    #melgram = np.load(str(train_clip) + '.npy')
    train_x = np.concatenate((train_x, melgram), axis=0)
os.chdir('/home/cc/notebooks/MusicProject/MagnaTagATune/')
np.save('train_x.npy', train_x)
print "Training file created."
