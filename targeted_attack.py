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
    access = 0
    
    # =====================================================================================
    # Step 1
    test_num = 50
    best_perturbed_image = None
    best_image_candidate = 1000*np.ones([test_num, 64, 64, 3], dtype=np.float32)
    best_l2_all = 1000
    for i in range(test_num):
        path = os.path.dirname(os.path.abspath(__file__))
        target_image = np.load(os.path.join(path, 'temp', "%03d"%label+"_"+str(i)+".npy"))
        target_image = target_image.astype(np.float32)

        access += 1
        if np.argmax(model.predictions(target_image)) != label:
            continue
        best_l2 = np.linalg.norm(image/255.0 - target_image/255.0)
        best_image_candidate[i] = copy.deepcopy(target_image)
        if best_l2 < best_l2_all:
            best_perturbed_image = copy.deepcopy(target_image)
            best_l2_all = best_l2

        noise = best_image_candidate[i] - image
        low, high = 0, 1
        while high-low >= 0.01:
            mid = (low+high)/2.0
            perturbed_image = image + mid * noise
            perturbed_image = np.round(perturbed_image.astype(np.float32))
            access += 1
            if np.argmax(model.predictions(perturbed_image)) == label:
                high = mid
                l2 = np.linalg.norm(image/255.0 - perturbed_image/255.0)
                if l2 < best_l2:
                    best_image_candidate[i] = copy.deepcopy(perturbed_image)
                    best_l2 = l2
                    if l2 < best_l2_all:
                        best_perturbed_image = copy.deepcopy(perturbed_image)
                        file_name = "%03d"%label+"_"+str(i)
                        best_l2_all = l2
            else:
                l2 = np.linalg.norm(image/255.0 - perturbed_image/255.0)
                if l2 > best_l2_all:
                    break
                low = mid
    
    if best_perturbed_image is None:
        return None


    # =====================================================================================
    # Step 2
    criterion = foolbox.criteria.TargetClass(label)
    attack = attacks.curls_targeted(model, criterion)
    for i in range(3):
        best_perturbed_image = attack(image, label, random_start=best_perturbed_image, scale=25, iterations=8, binary_search=7, return_early=True, 
                      epsilon=0.2, bb_step=3, RO=False, m=1, RC=False, TAP=False, uniform_or_not=False, moment_or_not=False)
        access = access + (8 + 3) * 7


    # =====================================================================================
    # Step 3
    noise = best_perturbed_image - image
    for i in range(255, 0, -1):
        if access > 500:
            break

        noise_temp = copy.deepcopy(noise)
        noise_temp[(noise_temp == i)] //= 2
        if (noise != noise_temp).any():
            access += 1
            if np.argmax(model.predictions(noise_temp + image)) == label:
                noise = copy.deepcopy(noise_temp)

        noise_temp = copy.deepcopy(noise)
        noise_temp[(noise_temp == -i)] //= 2
        if (noise != noise_temp).any():
            access += 1
            if np.argmax(model.predictions(noise_temp + image)) == label:
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

        np.random.seed(label+2)
        target_class = int(np.random.random()*200)
        adversarial = attack_func(model, image, target_class)
        store_adversarial(file_name, adversarial)

        
        if adversarial is None:
            print("can't find")
        elif np.argmax(model.predictions(adversarial)) == target_class:
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