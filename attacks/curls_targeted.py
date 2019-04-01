#coding=utf-8

from __future__ import division
import numpy as np
from abc import abstractmethod
import logging
import warnings

from foolbox import distances
from foolbox.utils import crossentropy

from foolbox.attacks.base import Attack
from foolbox.attacks.base import call_decorator

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


class IterativeProjectedGradientBaseAttack(Attack):
    @abstractmethod
    def _gradient(self, a, adv_x, class_, strict=True, x=None):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)
        self.success_dir = 0   #record of the average direction of all adversarial examples
        self.success_adv = 0

        self.best = 9999

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k, scale=scale, bb_step=bb_step, RO=RO, m=m, RC=RC, TAP=TAP, uniform_or_not=uniform_or_not, moment_or_not=moment_or_not)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, scale=scale, bb_step=bb_step, RO=RO, m=m, RC=RC, TAP=TAP, uniform_or_not=uniform_or_not, moment_or_not=moment_or_not)

    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not)
        
        if random_start is not None:
            a.predictions(random_start)

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
            else:
                bad = epsilon


    def update_success_dir(self, new_adv):
        self.success_adv += new_adv

        new_adv_norm = np.sqrt(np.mean(np.square(self.success_adv)))
        new_adv_norm = max(1e-12, new_adv_norm)
        self.success_dir = self.success_adv/new_adv_norm


    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early, scale, 
                 bb_step=15, RO=False, m=2, RC=False, TAP=False, uniform_or_not=False, moment_or_not=False):
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.original_image.copy()

        if random_start is not None:
            x = random_start
            strict = False

        if RC:   #use curl iteration to update adversarial example
            success = False
            momentum_up = 0
            momentum_down = 0
            go_up_flag = True   #gradient descend flag
            x_up = x.copy()

            logits_init, is_adversarial_init = a.predictions(np.round(x))
            ce_init = crossentropy(class_, logits_init)
            up_better_start = x.copy()

            for _ in range(iterations):
                avg_gradient_down = 0
                avg_gradient_up = 0
                for m_counter in range(m):
                    #gradient ascent trajectory
                    if RO:
                        if uniform_or_not:  #add uniform noise to gradient calculation process 
                            temp_x_up = np.clip(np.random.uniform(-scale, scale, original.shape) + x_up + stepsize*self.success_dir, min_, max_).astype(np.float32)
                        else:   #add gaussian noise to gradient calculation process
                            temp_x_up = np.clip(np.random.normal(loc=x_up, scale=scale) + stepsize*self.success_dir, min_, max_).astype(np.float32)
                    else:
                        if uniform_or_not:
                            temp_x_up = np.clip(np.random.uniform(-scale, scale, original.shape) + x_up, min_, max_).astype(np.float32)
                        else:
                            temp_x_up = np.clip(np.random.normal(loc=x_up, scale=scale), min_, max_).astype(np.float32)
                    temp_x_up.dtype = "float32"
                    gradient_up = self._gradient(a, temp_x_up, class_, strict=strict)   #calculate gradient on substitute model
                    avg_gradient_up += gradient_up

                    #gradient descent trajectory
                    if RO:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x + stepsize*self.success_dir, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale) + stepsize*self.success_dir, min_, max_).astype(np.float32)
                    else:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale), min_, max_).astype(np.float32)
                    temp_x_down.dtype = "float32"
                    gradient_down = self._gradient(a, temp_x_down, class_, strict=strict)
                    avg_gradient_down += gradient_down
                
                avg_gradient_up = avg_gradient_up/m
                avg_gradient_down = avg_gradient_down/m

                strict = True
                if targeted:
                    avg_gradient_down = -avg_gradient_down
                    avg_gradient_up = -avg_gradient_up

                if moment_or_not:   #whether use momentum as in MI-FGSM
                    momentum_up += avg_gradient_up
                    momentum_up_norm = np.sqrt(np.mean(np.square(momentum_up)))
                    momentum_up_norm = max(1e-12, momentum_up_norm)  # avoid divsion by zero

                    momentum_down += avg_gradient_down
                    momentum_down_norm = np.sqrt(np.mean(np.square(momentum_down)))
                    momentum_down_norm = max(1e-12, momentum_down_norm)  # avoid divsion by zero
                    if go_up_flag:
                        x_up = x_up - stepsize * (momentum_up/momentum_up_norm)
                    else:
                        x_up = x_up + stepsize * (momentum_up/momentum_up_norm)

                    x = x + stepsize * (momentum_down/momentum_down_norm)

                else: 
                    if go_up_flag:
                        avg_gradient_up = -avg_gradient_up
                        x_up = x_up + stepsize * avg_gradient_up
                    else:
                         x_up = x_up + stepsize * avg_gradient_up

                    x = x + stepsize * avg_gradient_down

                x = original + self._clip_perturbation(a, x - original, epsilon)
                x_up = original + self._clip_perturbation(a, x_up - original, epsilon)

                x = np.clip(x, min_, max_)
                x_up = np.clip(x_up, min_, max_)

                logits_down, is_adversarial_down = a.predictions(np.round(x))
                logits_up, is_adversarial_up = a.predictions(np.round(x_up))

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits_down)
                        logging.debug('crossentropy to {} is {}'.format(
                            a.original_class, ce))
                    ce = crossentropy(class_, logits_down)
                    logging.debug('crossentropy to {} is {}'.format(class_, ce))

                if is_adversarial_up:
                    if RO:
                        self.update_success_dir(x_up)
                    #start binary search
                    left = original
                    right = x_up
                    for binary_counter in range(bb_step):
                        middle = np.clip((left + right)/2, min_, max_)
                        temp_logits, temp_is_adversarial = a.predictions(np.round(middle))

                        if temp_is_adversarial: #find a better adversarial example
                            if RO:
                                self.update_success_dir(middle)
                            right = middle
                        else:
                            left = middle
                    if return_early:
                        return True
                    else:
                        success = True

                if is_adversarial_down:
                    if RO:
                        self.update_success_dir(x)
                    left = original
                    right = x
                    for binary_counter in range(bb_step):
                        middle = np.clip((left + right)/2, min_, max_)
                        temp_logits, temp_is_adversarial = a.predictions(np.round(middle))

                        if temp_is_adversarial:
                            if RO:
                                self.update_success_dir(middle)
                            right = middle
                        else:
                            left = middle
                    if return_early:
                        return True
                    else:
                        success = True

    
                if go_up_flag:
                    ce_now = crossentropy(class_, logits_up)
                    if ce_now < ce_init:
                        ce_init = ce_now
                        up_better_start = x_up
                    else:
                        go_up_flag = False    #stop gradient descent, start gradient ascent
                        momentum_up = 0
                        x_up = up_better_start


        else:    #normal iterative process
            success = False
            momentum_down = 0

            for _ in range(iterations):
                avg_gradient_down = 0
                avg_gradient_up = 0
                for m_counter in range(m):
                    if RO:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x + stepsize*self.success_dir, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale) + stepsize*self.success_dir, min_, max_).astype(np.float32)
                    else:
                        if uniform_or_not:
                            temp_x_down = np.clip(np.random.uniform(-scale, scale, original.shape) + x, min_, max_).astype(np.float32)
                        else:
                            temp_x_down = np.clip(np.random.normal(loc=x, scale=scale), min_, max_).astype(np.float32)
                    temp_x_down.dtype = "float32"
                    gradient_down = self._gradient(a, temp_x_down, class_, strict=strict)
                    avg_gradient_down += gradient_down
                
                avg_gradient_down = avg_gradient_down/m

                strict = True
                if targeted:
                    avg_gradient_down = -avg_gradient_down

                if moment_or_not:
                    momentum_down += avg_gradient_down
                    momentum_down_norm = np.sqrt(np.mean(np.square(momentum_down)))
                    momentum_down_norm = max(1e-12, momentum_down_norm)  # avoid divsion by zero
                    x = x + stepsize * (momentum_down/momentum_down_norm)

                else: 
                    x = x + stepsize * avg_gradient_down

                x = original + self._clip_perturbation(a, x - original, epsilon)
                x = np.clip(x, min_, max_)

                logits_down, is_adversarial_down = a.predictions(np.round(x))

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    if targeted:
                        ce = crossentropy(a.original_class, logits_down)
                        logging.debug('crossentropy to {} is {}'.format(
                            a.original_class, ce))
                    ce = crossentropy(class_, logits_down)
                    logging.debug('crossentropy to {} is {}'.format(class_, ce))

                if is_adversarial_down:
                    if RO:
                        self.update_success_dir(x)
                    left = original
                    right = x
                    for binary_counter in range(bb_step):
                        middle = np.clip((left + right)/2, min_, max_)
                        temp_logits, temp_is_adversarial = a.predictions(np.round(middle))

                        if temp_is_adversarial: 
                            if RO:
                                self.update_success_dir(middle)
                            right = middle
                        else:
                            left = middle
                    if return_early:
                        return True
                    else:
                        success = True
        return success
        


class L2GradientMixin(object):
    def _gradient(self, a, adv_x, class_, strict=True, x=None):
        if x is None:
            gradient = a.gradient(adv_x, class_, strict=strict)
        else:
            gradient = a.gradient(x, adv_x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(max(1e-12, np.mean(np.square(gradient))))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient



class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor



class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L2 norm of the perturbation without'
                            ' specifying foolbox.distances.MSE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')



class L2BasicIterativeAttack(
        L2GradientMixin,
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):

    """Modified version of the Basic Iterative Method
    that minimizes the L2 distance.

    .. seealso:: :class:`LinfinityBasicIterativeAttack`

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True,
                 scale = 2,
                 bb_step = 10,
                 RO = False, 
                 m=1, 
                 RC=False, 
                 TAP=False, 
                 uniform_or_not=False, 
                 moment_or_not=False):

        """Simple iterative gradient-based attack known as
        Basic Iterative Method, Projected Gradient Descent or FGSM^k.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool or int
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        random_start : `numpy.ndarray`
            Start the attack from an adversarial image.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        scale : float
            Variance of gaussian noise add to gradient calculation.
        bb_step : int
            Binary search step after each round of iteration.
        RO : bool
            Whether to update initial direction by the average direction
            of all adversarial examples.
        m : int
            Times of gradient calculation as mentioned in vr-IGSM attack.
        RC : bool
            Whether use Curls iteration to update adversarial trajectory.
        TAP : bool
            Discarded parameter.
        Uniform_or_not : bool
            Whether use uniform noise (if set to True) or gaussian noise (if
            set to False).
        moment_or_not : bool
            Whether use momentum for update in iteration. 
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early, scale, bb_step, RO, m, RC, TAP, uniform_or_not, moment_or_not)






