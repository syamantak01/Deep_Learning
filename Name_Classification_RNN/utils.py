import torch
import random
import io 
import os 
import unicodedata
import string
import glob

#alphabets(lowercase+uppercase) + '.,;'
characters = string.ascii_letters + " .,;'"
n_char = len(characters) 

#convert unicode string to its subsequent ASCII representation(we also have to remove the accents)
#https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string/518232#518232
def convert_to_ascii(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn' and c in characters)

def load_data(): 
    names_by_language = {} #stores the list of names for every language 
    languages = [] #list of all the languages 


    #read a file and split by new line
    def read_lines(file):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            names = [convert_to_ascii(line) for line in lines]

        return names

    #find the file path of the files with a .txt extension 
    find_file = lambda path: glob.glob(path)

    for file in find_file('data/*.txt'):
        language = os.path.splitext(os.path.basename(file))[0]
        languages.append(language)

        names = read_lines(file)
        names_by_language[language] = names

    return names_by_language, languages
    
            
#Letters to be represented as 'one-hot encoded vectors' of the shape (1, n_char)
#And names are represented as a matrix of stacked one-hot encoded vectors(which are representation of the character/letterrs)
#shape of name-matrix will be (number_of_characters_in_a_word, 1, n_char); 1 denotes the batch size, which is important in PyTorch representation.

#converts a char into an one-hot encoded vector(tensor)
def char_to_tensor(c):
    char_tensor = torch.zeros(1, n_char)
    index = characters.find(c)
    char_tensor[0][index] = 1
    return char_tensor

#converts names to a tensor
def name_to_tensor(name):
    name_tensor = torch.zeros(len(name), 1, n_char)
    for i, char in enumerate(name):
        name_tensor[i][0] = char_to_tensor(char)

    return name_tensor

def random_training_example(names_by_language, languages):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    language = random_choice(languages)
    name = random_choice(names_by_language[language])
    language_tensor = torch.tensor([languages.index(language)], dtype=torch.long)
    name_tensor = name_to_tensor(name)
    return language, name, language_tensor, name_tensor

# if __name__ == "__main__":
#     print(characters)
#     print(convert_to_ascii('Ślusàrski'))

#     names_by_language, languages = load_data()
#     print(names_by_language['Italian'][:5])

#     print(char_to_tensor('J'))
#     print(name_to_tensor('Jones').size())