import pickle
import os
import pdb
a_file = open("label_to_class.pkl", "rb")
output = pickle.load(a_file)
print(output[0])

for img in os.listdir("grad_cam_results_before_posttraining"):
    #pdb.set_trace()
    if "_class_" in img:
        class_png = img.split("_class_")[1]
        before = img.split("_class_")[0]
        class_only = class_png.split(".")[0]
        thestring = output[int(class_only)]['class']
        thestring = thestring.replace(" ", "_")
        thestring = thestring.replace(",", "_")
        finalname = before + "_" + thestring
        endresult = os.path.join(os.path.dirname("grad_cam_results_before_posttraining/" + img), finalname)
        os.rename("grad_cam_results_before_posttraining/" + img, endresult)

