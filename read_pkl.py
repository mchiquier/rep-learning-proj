import pickle

a_file = open("label_to_class.pkl", "rb")
output = pickle.load(a_file)
print(output[0])

