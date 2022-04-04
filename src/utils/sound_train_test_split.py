import os
from shutil import copyfile

# As per README:
# All files of iteration 0-4 move to testsing-spectrograms
# All files of iteration 5-49 move to training-spectrograms

def separate(source):
    
    num_list = sorted(os.listdir(source))
    # print(num_list)

    for i in num_list[:10]:
        sounds = os.listdir(os.path.join(source+ i))
        for j in sounds:
            first_split = j.rsplit("_", 1)[1]
            second_split = first_split.rsplit(".", 1)[0]
            if int(second_split) < 15: # index started from 0
                copyfile(source + i+'/' + j, "../data/sound_450/test/" + i + "/" + j)
            else:
                copyfile(source + i +'/' + j, "../data/sound_450/train/" + i + "/" + j)

separate("../data/sound/")