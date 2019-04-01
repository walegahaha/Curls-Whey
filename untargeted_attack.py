import os
import copy
import shutil
import attacks
import foolbox
import numpy as np
from fmodel import create_fmodel
from bmodel import create_bmodel
from utils import read_images, store_adversarial, compute_MAD

test_model_acc = False


def run_attack_curls_whey(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    attack = attacks.curls_untargeted(model, criterion)
    perturbed_image = attack(image, label, scale=1, iterations=5, binary_search=12, 
                             return_early=True, epsilon=0.3, bb_step=3, RO=True, 
                             m=1, RC=True, TAP=False, uniform_or_not=False, moment_or_not=False)
    
    if perturbed_image is None: return None

    noise = perturbed_image - image
    for i in range(255, 0, -1):
        noise_temp = copy.deepcopy(noise)
        noise_temp[(noise_temp == i)] //= 2
        if (noise != noise_temp).any():
            if np.argmax(model.predictions(noise_temp + image)) != label:
                noise = copy.deepcopy(noise_temp)

        noise_temp = copy.deepcopy(noise)
        noise_temp[(noise_temp == -i)] //= 2
        if (noise != noise_temp).any():
            if np.argmax(model.predictions(noise_temp + image)) != label:
                noise = copy.deepcopy(noise_temp)
     
    perturbed_image = noise + image
    
    return perturbed_image


def test(model, attack_func, method_name):
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results");

    acc = 0
    for (file_name, image, label) in read_images():  
        if test_model_acc == True:
            acc += np.argmax(model.predictions(image)) == label
            continue 
        print(file_name, end="\t\t")
        adversarial = attack_func(model, image, label)
        store_adversarial(file_name, adversarial)

        
        if adversarial is None:
            print("can't find")
        elif np.argmax(model.predictions(adversarial)) != label:
            print("l2: %.4f" %np.linalg.norm(image/255 - adversarial/255))
        else:
            print("error");
            exit()
        

    if test_model_acc == True:
        print("model accuracy:  %.4f" %(acc/200)); exit()
    
    print("\n", method_name, "\n")
    compute_MAD()


def main(): 
    forward_model = create_fmodel()
    backward_model = create_bmodel()
    
    model = foolbox.models.CompositeModel(
        forward_model=forward_model,
        backward_model=backward_model)

    print("\n\nStart Test...")
    test(model, run_attack_curls_whey, "Curls & Whey")


if __name__ == '__main__':
    main()