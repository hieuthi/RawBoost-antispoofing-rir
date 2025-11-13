
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import pandas
import numpy as np


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss       Cost of tandem system falsely rejecting target speaker.
                          Cfa         Cost of tandem system falsely accepting nontarget speaker.
                          Cfa_spoof   Cost of tandem system falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """


    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;


    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds

def compute_tDCF_legacy(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """


    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds



def load_asv_metrics(tar_asv, non_asv, spoof_asv):
    """ Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(
           tar_asv, non_asv, spoof_asv)
    input
    -----
      tar_asv    np.array, score of target speaker trials
      non_asv    np.array, score of non-target speaker trials
      spoof_asv  np.array, score of spoofed trials
    
    output
    ------
      Pfa_asv           scalar, value of ASV false accept rate
      Pmiss_asv         scalar, value of ASV miss rate
      Pmiss_spoof_asv   scalar, 
      P_fa_spoof_asv    scalar
    """
    # 
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold)
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv




def get_eer(bonafide_score_cm, spoof_score_cm):
    """ eer_val, threshold = get_eer(bonafide_score_cm, spoof_score_cm)

    input
    -----
      bonafide_score_cm np.array, score of bonafide data
      spoof_score_cm    np.array, score of spoofed data
    
    output
    ------
      eer_val           scalar, value of EER
      threshold         scalar, value of the threshold corresponding to EER
    """
    eer_val, threshold = compute_eer(bonafide_score_cm, spoof_score_cm)
    return eer_val, threshold


def get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model):
    """C0, C1, C2 = get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)
    
    compute_tDCF can be factorized into two parts: 
    C012 computation and min t-DCF computation.

    This is for C012 computation.
    
    input
    -----
      Pfa_asv           scalar, value of ASV false accept rate
      Pmiss_asv         scalar, value of ASV miss rate
      Pmiss_spoof_asv   scalar, 
      P_fa_spoof_asv    scalar
      
    output
    ------
      C0                 scalar, coefficient for min tDCF computation
      C1                 scalar, coefficient for min tDCF computation
      C2                 scalar, coefficient for min tDCF computation
    
    """
    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')
        sys.exit(1)

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or \
       cost_model['Pspoof'] < 0 or \
       np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        print('ERROR: Your prior probabilities should be positive and sum up to one.')
        sys.exit(1)

    # Unless we evaluate worst-case model, 
    # we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        print('ERROR: please provide false alarm rate of spoof tests against ASV system.')
        sys.exit(1)

    
    # Constants - see ASVspoof 2019/21 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv \
         + cost_model['Pnon'] * cost_model['Cfa'] *Pfa_asv
    
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] \
         - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv \
            + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;
    
    return C0, C1, C2


def get_tDCF_C012_from_asv_scores(tar_asv, non_asv, spoof_asv, cost_model):
    """ C0, C1, C2 = get_tDCF_C012_from_asv_scores(tar_asv, non_asv, spoof_asv, cos_model)

    Wrapper combining load_asv_metrics and get_tDCF_C012.

    input
    -----
      tar_asv    np.array, score of target speaker trials
      non_asv    np.array, score of non-target speaker trials
      spoof_asv  np.array, score of spoofed trials

    output
    ------
      C0                 scalar, coefficient for min tDCF computation
      C1                 scalar, coefficient for min tDCF computation
      C2                 scalar, coefficient for min tDCF computation
    
    For C0, C1, C2, see Appendix Eqs.(1-2) in evaluation plan [1],
    or Eqs.(10-11) in [2]

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """
    # here we only consider the case where tar, nontarget, and spoof scores
    # are all avaialble
    if len(tar_asv) and len(non_asv) and len(spoof_asv):
        # compute ASV metrics
        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(
            tar_asv, non_asv, spoof_asv)

        # get the C012 values
        C0, C1, C2 = get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)
    else:
        C0, C1, C2 = np.nan, np.nan, np.nan

    return C0, C1, C2


def get_mintDCF_eer(bonafide_score_cm, spoof_score_cm, C0, C1, C2):
    """ mintDCF, eer = get_mintDCF_eer(bonafide_score_cm, spoof_score_cm, C0, C1, C2)

    input
    -----
      bonafide_score_cm  np.array, score of bonafide data
      spoof_score_cm     np.array, score of spoofed data
      C0                 scalar, coefficient for min tDCF computation
      C1                 scalar, coefficient for min tDCF computation
      C2                 scalar, coefficient for min tDCF computation
    
    output
    ------
      eer                scalar, value of EER
      mintDCF            scalar, value of min tDCF

    For C0, C1, C2, see Appendix Eqs.(1-2) in evaluation plan [1],
    or Eqs.(10-11) in [2]

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """
    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    mintDCF = tDCF_norm[tDCF_norm.argmin()]

    abs_diffs = np.abs(Pmiss_cm - Pfa_cm)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((Pmiss_cm[min_index], Pfa_cm[min_index]))

    return mintDCF, eer, CM_thresholds[min_index]


def dump_C012_dict(data_dict, filepath):
    np.array(data_dict).dump(filepath)
    return 

def load_C012_dict(filepath):
    return dict(np.load(filepath, allow_pickle=True).tolist())


def load_C012_value(C012_buf, factor_list, fail_value = np.nan):
    """ C0, C1, C2 = load_C012_value(C012_buf, factor_list)
    input
    -----
      C012_buf     dictionary, value of C012 in dictionary
      factor_list  list of str, list of factors to retrive the value.
                   value is given by C012_buf[factor_list[0]][factor_list[1]]...
    output
    ------
      C0           scalar
      C1           scalar
      C2           scalar
    """
    tmp = C012_buf
    try:
        for factor in factor_list:
            tmp = tmp[factor]
        C0, C1, C2 = tmp['C0'], tmp['C1'], tmp['C2']
    except KeyError:
        # cannot load C012 from the dictionary
        C0, C1, C2 = np.nan, np.nan, np.nan
    return C0, C1, C2

def save_C012_value(C012_buf, C0, C1, C2, factor_list):
    """
    """
    if not type(C012_buf) is dict:
        print("C012_buf is not a dictionary")
        sys.exit(1)
    else:
        tmp = C012_buf
        try:
            for factor in factor_list:
                if not factor in tmp:
                    tmp[factor] = dict()
                tmp = tmp[factor]
            tmp['C0'], tmp['C1'], tmp['C2'] = C0, C1, C2
        except KeyError:
            print("Fail to push C012 to dictionary")
            sys.exit(1)
    return

def load_protocol(protocol_file, names, sep=' ', index_col=None):
    """ pd_protocol = load_protocol(protocol_file, names, sep=' ', index_col=None)

    input
    -----
      protocol_file  str, path to the protocol file
      names          list of str, name of the data Series in dataFrame 
      sep            str, separator, by default ' '
      index_col      str, name of the index column, by default None

    output
    ------
      pd_protocol    pandas dataFrame
    """
    pd_protocol = pandas.read_csv(protocol_file, sep=' ', names=names, 
                                  index_col = index_col, skipinitialspace=True)
    return pd_protocol


def load_score(score_file,  names,  sep=' ', index_col=None):
    """ pd_score = load_score(score_file, names, sep=' ', index_col=None)                                 
                                                                                                          
    input                                                                                                 
    -----                                                                                                 
      score_file     str, path to the score file                                                          
      names          list of str, name of the data Series in dataFrame                                    
      sep            str, separator, by default ' '                                                       
      index_col      str, name of the index column, by default None                                       
                                                                                                          
    output                                                                                                
    ------                                                                                                
      pd_protocol    pandas dataFrame                                                                     
    """
    pd_score = pandas.read_csv(score_file, sep=sep, names=names,
                                  index_col=index_col, skipinitialspace=True)
    return pd_score

def join_protocol_score(pd_protocol, pd_score):
    """ pd_final = join_protocol_score(pd_protocol, pd_score)                                             
                                                                                                          
    input                                                                                                 
    -----                                                                                                 
      pd_protocol  dataFrame, protocol dataframe                                                          
      pd_score     dataFrame, score dataframe                                                             
                                                                                                          
    output                                                                                                
    ------                                                                                                
      pd_final     dataFrame, joint dataFrame from pd_protocol and pd_score                               
    """
    pd_final = pandas.concat([pd_protocol, pd_score], axis=1, join="inner")
    if len(pd_protocol) != len(pd_score) or len(pd_protocol) != len(pd_final):
        print("Error: protocol and score seem to mismatch. Please check!")
        print("Protocol file has {:d} entries".format(len(pd_protocol)))
        print("Score file has {:d} entries".format(len(pd_score)))
        print("Number of common entries is {:d}".format(len(pd_final)))
        print("\nIs the score file incomplete?")
        print("Has you selected the correct track?")
        sys.exit(1)
    return pd_final

g_factor_type_spoof = 'spoof'
g_factor_type_bonafide = 'bonafide'
g_factor_type_both = 'both'

g_bonafide_tag = 'bonafide'
g_spoofed_tag = 'spoof'
g_score_col_name = 'score'
g_pooled_tag = 'Pooled'

g_target_tag = 'target'
g_nontarget_tag = 'nontarget'

g_LA_track = 'LA'
g_DF_track = 'DF'
g_PA_track = 'PA'

g_possible_subsets = ['eval', 'progress', 'hidden', 'hidden1_PA', 'hidden2_PA']
g_possible_tracks = ['LA', 'DF', 'PA']


def compute_decomposed_mintdcf_eer(score_pd, 
                                   factor_name_v,
                                   factor_value_v, 
                                   factor_type_v,
                                   factor_name_h, 
                                   factor_value_h, 
                                   factor_type_h,
                                   C012_buf = None,
                                   pooled_tag = g_pooled_tag, 
                                   bonafide_tag = g_bonafide_tag,
                                   spoofed_tag = g_spoofed_tag,
                                   col_score_name = g_score_col_name,
                                   flag_verbose = False):
    """mintDCF_array, eer_array = compute_decomposed_mintdcf_eer(score_pd, 
                                   factor_name_v,
                                   factor_value_v, 
                                   factor_type_v,
                                   factor_name_h,
                                   factor_value_h,  
                                   factor_type_h,
                                   C012_buf = None,
                                   pooled_tag = 'Pooled', 
                                   bonafide_tag = 'bonafide',
                                   spoofed_tag = 'spoof',
                                   col_score_name = 'score',
                                   flag_verbose = False)
    
    Function to loop over two sets of factors and compute min t-DCF and EER in
    each pair of the factor.

    input
    -----
      score_pd        dataFrame, joint dataframe of CM score and protocol

      factor_name_v   str or list of str, 
                      name(s) of the dataFrame series for the 1st set of factor.
                    
      factor_value_v  list of str, or list of list or str, 
                      values of the 1st set of factors 

                      if type(factor_name_v) is str:
                          # we retrieve the data by
                          for factor in factor_value_v:
                              data = score_pd.query('factor_name_v == "factor"')

                      if type(factor_name_v) is list
                          # we iterate all the factors
                          for factor_name, factor_value in zip(factor_name_v, factor_value_v):
                              for factor in factor_value:
                                  data = score_pd.query('factor_name == "factor"')
 
                    
                      The second case is useful when the 1st set of factors 
                      are defined in different data series of score_pd.

      factor_type_v   str or list of str, type of the factor
                      
                      'spoof': this factor is only available for spoofed data
                      'bonafide': this factor is only available for bonafide data
                      'both': this factor appears in both spoofed and bonafide data

                      if type(factor_name_v) is str:
                          # factor_type_v is the type for factor_name_v
                      if type(factor_name_v) is list:
                          # factor_type_v should be a list and
                          # factor_type_v[i] is the type for factor_name_v[i]

      factor_name_h   str or list of str, 
      factor_value_h  list of str or list of list of str
      factor_type_h   str or list of str
                     
                      these are for the second set of factors

      C012_buf        dict, dictionary of C0, C1, C2 values for each condition
                      we will use load_C012_value(C012_buf, [factor1, factor2])
                      to load the C0, C1, C2 value

                      if C012_buf is None, mintDCF_array will be [np.nan]

      pooled_tag      str, tag for pooled condition, 
                      default 'Pooled'
      bonafide_tag    str, tag for bonafide trials
                      default 'bonafide'
      spooed_tag      str, tag for spoofed trials
                      default 'spoof'
      col_score_name  str, name of the column for score
                      default 'score'

    output
    ------
      mintDCF_array   np.array, min t-DCF values in all conditions
                      mintDCF_array.shape[0] is equal to the length of all
                      possible values in factor_value_v
                      mintDCF_array.shape[1] is equal to the length of all
                      possible values in factor_value_h

      eer_array       np.array, EER values in all conditions.
                      same shape as mintDCF_array
    """
    def _wrap_list(data):
         return [data] if type(data) is str else data

    def _wrap_list_list(data):
         return [data] if type(data[0]) is str else data
    
    # wrap them into a list
    factor_names_1 = _wrap_list(factor_name_v)
    factor_names_2 = _wrap_list(factor_name_h)
    factor_types_1 = _wrap_list(factor_type_v)
    factor_types_2 = _wrap_list(factor_type_h)
    factor_list_1_list = _wrap_list_list(factor_value_v)
    factor_list_2_list = _wrap_list_list(factor_value_h)
        
    # number of rows and columns in the result table
    num_row = sum([len(x) for x in factor_list_1_list])
    num_col = sum([len(x) for x in factor_list_2_list])

    # output buffer
    mintDCF_array = np.zeros([num_row, num_col])
    eer_array = np.zeros_like(mintDCF_array)
    thres_array = np.zeros_like(mintDCF_array)

    if C012_buf is None:
        print('\n' + ''.join(['-'] * (num_row - 1)) + '>| computing EERs')
    else:
        print('\n' + ''.join(['-'] * (num_row - 1)) + '>| computing EERs and min tDCF')

    # loop over factor along the row (factor 1)
    id1 = 0
    for factor_name_1, factor_list_1, factor_type_1 in zip(
        factor_names_1, factor_list_1_list, factor_types_1):
        for _, factor_1 in enumerate(factor_list_1):
            
            print(".", end = '', flush=True)
            
            # creat the query to retrieve the data corresponding to the factor_1
            if factor_1 == pooled_tag:
                # pooled condition
                qr_bona_fac1 = ''
                qr_spoof_fac1 = ''
            elif factor_type_1 == g_factor_type_spoof:
                # if factor is only for spoofed data (e.g., attack type)
                qr_bona_fac1 = ''
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            elif factor_type_1 == g_factor_type_bonafide:
                # if factor is only for bonafide
                qr_bona_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ''
            else:
                # if factor is for both spoofed and bona fide data (e.g., codec)
                qr_bona_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            
            # loop over factor in cols (factor 2)
            id2 = 0
            for factor_name_2, factor_list_2, factor_type_2 in zip(
                factor_names_2, factor_list_2_list, factor_types_2):
                for _, factor_2 in enumerate(factor_list_2):
                    
                    if factor_2 == pooled_tag:
                        # pooled condition
                        qr_bona_fac2 = ''
                        qr_spoof_fac2 = ''
                    elif factor_type_2 == g_factor_type_spoof:
                        # if factor is only for spoofed data (e.g., attack type)
                        qr_bona_fac2 = ''
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                    elif factor_type_2 == g_factor_type_bonafide:
                        qr_bona_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ''
                    else:
                        # if factor is for both spoofed and bona fide data (e.g., codec)
                        qr_bona_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)

                    # query that we will use to retrieve the data
                    qr_bonafide = 'label == "{:s}"'.format(bonafide_tag) + qr_bona_fac1 + qr_bona_fac2
                    qr_spoof = 'label == "{:s}"'.format(spoofed_tag) +  qr_spoof_fac1 + qr_spoof_fac2
                    
                    # retrive data
                    bona_env_pd = score_pd.query(qr_bonafide)
                    spoof_env_pd = score_pd.query(qr_spoof)
                    bona_data = bona_env_pd[col_score_name].to_numpy()
                    spoof_data = spoof_env_pd[col_score_name].to_numpy()
                    
                    # load C012 values
                    if C012_buf is None:
                        # dummy value 
                        C0, C1, C2 = 0.1, 0.1, 0.1
                    else:
                        C0, C1, C2 = load_C012_value(
                            C012_buf, [factor_1, factor_2])

                    # print infor
                    if flag_verbose:
                        print(qr_bonafide, "{:d} entries".format(len(bona_data)))
                        print(qr_spoof, "{:d} entries".format(len(spoof_data)))
                    
                    # computation
                    if len(bona_data) and len(spoof_data):
                        mintdcf, eer_tmp, thres_tmp = get_mintDCF_eer(
                            bona_data, spoof_data, C0, C1, C2)
                    else:
                        mintdcf, eer_tmp, thres_tmp = np.nan, np.nan, np.nan

                    # mask the min t-DCF values when C012 is invalid
                    if C012_buf is None:
                        mintdcf = mintdcf * np.nan
                        
                    # save the value
                    mintDCF_array[id1, id2] = mintdcf
                    eer_array[id1, id2] = eer_tmp
                    thres_array[id1, id2] = thres_tmp

                    id2 += 1
            # loop over horizotal factors
            id1 += 1
    print("")
    return mintDCF_array, eer_array, thres_array
# 
Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
}

# function to compute tDCF C012 coeffcients over two sets of factors
def compute_tDCF_C012(asv_score_pd, 
                      factor_name_v,
                      factor_value_v, 
                      factor_type_v,
                      factor_name_h, 
                      factor_value_h, 
                      factor_type_h,
                      cost_model = cost_model,
                      pooled_tag = g_pooled_tag, 
                      target_tag = g_target_tag,
                      nontarget_tag =g_nontarget_tag,
                      spoofed_tag = g_spoofed_tag,
                      col_score_name = g_score_col_name,
                      flag_verbose = False):
    """C012_dict = compute_tDCF_C012(asv_score_pd, 
                                   factor_name_v,
                                   factor_value_v, 
                                   factor_type_v,
                                   factor_name_h,
                                   factor_value_h,  
                                   factor_type_h,
                                   cost_model = cost_model,
                                   pooled_tag = 'Pooled', 
                                   target_tag = 'target',
                                   nontarget_tag = 'nontarget',
                                   spoofed_tag = 'spoof',
                                   col_score_name = 'score',
                                   flag_verbose = False)
    
    Function to loop over two sets of factors and compute C012.
    The output C012_dict can be used to compute min tDCF values

    input
    -----
      asv_score_pd    dataFrame, joint dataframe of ASV score and protocol

      factor_name_v   str or list of str, 
                      name(s) of the dataFrame series for the 1st set of factor.
                    
      factor_value_v  list of str, or list of list or str, 
                      values of the 1st set of factors 

                      if type(factor_name_v) is str:
                          # we retrieve the data by
                          for factor in factor_value_v:
                              data = score_pd.query('factor_name_v == "factor"')

                      if type(factor_name_v) is list
                          # we iterate all the factors
                          for factor_name, factor_value in zip(factor_name_v, factor_value_v):
                              for factor in factor_value:
                                  data = score_pd.query('factor_name == "factor"')
 
                    
                      The second case is useful when the 1st set of factors 
                      are defined in different data series of score_pd.

      factor_type_v   str or list of str, type of the factor
                      
                      'spoof': this factor is only available for spoofed data
                      'bonafide': this factor is only available for bonafide data
                      'both': this factor appears in both spoofed and bonafide data

                      if type(factor_name_v) is str:
                          # factor_type_v is the type for factor_name_v
                      if type(factor_name_v) is list:
                          # factor_type_v should be a list and
                          # factor_type_v[i] is the type for factor_name_v[i]

      factor_name_h   str or list of str, 
      factor_value_h  list of str or list of list of str
      factor_type_h   str or list of str
                     
                      these are for the second set of factors

      pooled_tag      str, tag for pooled condition, 
                      default 'Pooled'
      target_tag      str, tag for bonafide tareget trials
                      default 'target'
      nontarget_tag   str, tag for bonafide non-tareget trials
                      default 'nontarget'
      spooed_tag      str, tag for spoofed trials
                      default 'spoof'
      col_score_name  str, name of the column for score
                      default 'score'

    output
    ------
      C012_dict       dictionary of C012 values
                      C012[factor_1][factor_2]['C0'] -> C0
                      C012[factor_1][factor_2]['C1'] -> C1
                      C012[factor_1][factor_2]['C2'] -> C2
    """
    def _wrap_list(data):
         return [data] if type(data) is str else data

    def _wrap_list_list(data):
         return [data] if type(data[0]) is str else data
    
    # wrap them into a list
    factor_names_1 = _wrap_list(factor_name_v)
    factor_names_2 = _wrap_list(factor_name_h)
    factor_types_1 = _wrap_list(factor_type_v)
    factor_types_2 = _wrap_list(factor_type_h)
    factor_list_1_list = _wrap_list_list(factor_value_v)
    factor_list_2_list = _wrap_list_list(factor_value_h)

    # number of rows and columns in the result table
    num_row = sum([len(x) for x in factor_list_1_list])
    num_col = sum([len(x) for x in factor_list_2_list])

    # output buffer
    C012 = dict()

    print('\n' + ''.join(['-'] * (num_row - 1)) + '>| computing C012 for tDCF')

    # loop over factor along the row (factor 1)
    for factor_name_1, factor_list_1, factor_type_1 in zip(
        factor_names_1, factor_list_1_list, factor_types_1):
        for _, factor_1 in enumerate(factor_list_1):
            
            print(".", end = '', flush=True)
            
            # creat the query to retrieve the data corresponding to the factor_1
            if factor_1 == pooled_tag:
                # pooled condition
                qr_tar_fac1 = ''
                qr_ntar_fac1 = ''
                qr_spoof_fac1 = ''
            elif factor_type_1 == g_factor_type_spoof:
                # if factor is only for spoofed data (e.g., attack type)
                qr_tar_fac1 = ''
                qr_ntar_fac1 = ''
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            elif factor_type_1 == g_factor_type_bonafide:
                # if factor is only for bonafide (target and nontarget)
                qr_tar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_ntar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ''
            else:
                # if factor is for both spoofed and bona fide data (e.g., codec)
                qr_tar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_ntar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            
            # loop over factor in cols (factor 2)
            for factor_name_2, factor_list_2, factor_type_2 in zip(
                factor_names_2, factor_list_2_list, factor_types_2):
                for _, factor_2 in enumerate(factor_list_2):
                                        
                    if factor_2 == pooled_tag:
                        # pooled condition
                        qr_tar_fac2 = ''
                        qr_ntar_fac2 = ''
                        qr_spoof_fac2 = ''
                    elif factor_type_2 == g_factor_type_spoof:
                        # if factor is only for spoofed data (e.g., attack type)
                        qr_tar_fac2 = ''
                        qr_ntar_fac2 = ''
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                    elif factor_type_2 == g_factor_type_bonafide:
                        qr_tar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_ntar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ''
                    else:
                        # if factor is for both spoofed and bona fide data (e.g., codec)
                        qr_tar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_ntar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)

                    # query that we will use to retrieve the data
                    qr_tar = 'label == "{:s}"'.format(target_tag) + qr_tar_fac1 + qr_tar_fac2
                    qr_ntar = 'label == "{:s}"'.format(nontarget_tag) + qr_ntar_fac1 + qr_ntar_fac2
                    qr_spoof = 'label == "{:s}"'.format(spoofed_tag) +  qr_spoof_fac1 + qr_spoof_fac2
                    
                    # retrive data
                    tar_env_pd = asv_score_pd.query(qr_tar)
                    ntar_env_pd = asv_score_pd.query(qr_ntar)
                    spoof_env_pd = asv_score_pd.query(qr_spoof)
                    
                    tar_data = tar_env_pd[col_score_name].to_numpy()
                    ntar_data = ntar_env_pd[col_score_name].to_numpy()
                    spoof_data = spoof_env_pd[col_score_name].to_numpy()
                    
                    # compute C012 values
                    C0, C1, C2 = get_tDCF_C012_from_asv_scores(
                        tar_data, ntar_data, spoof_data, cost_model)
                    # save C012 coef
                    save_C012_value(
                        C012, C0, C1, C2, [factor_1, factor_2])
    print("")
    return C012
import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#####################
## Latex table
#####################
def return_one_row_latex(content_buffer):
    return " & ".join(content_buffer) + r"\\ " + "\n"
        
def return_one_row_text(content_buffer):
    return " ".join(content_buffer) + "\n"

def fill_cell(text, length, sep=''):
    return "{str:^{wid}}".format(str=text, wid=length) + sep
    
def wrap_value(data, wrap_factor=0):
    if wrap_factor == 0:
        return data
    else:
        ratio = (1+wrap_factor) / (1-wrap_factor)
        return np.power((1 - np.power(1 - data, ratio)), 1/ratio)

def return_latex_color_cell(value, val_min, val_max, scale, wrap, color_func):
    
    # clip the value for color rendering
    value = np.clip(value, val_min, val_max)
    
    # normalized value
    if scale < 0:
        value = wrap_value((value - val_min) / (val_max - val_min), wrap)*-scale
        value = -scale - value
    else:
        value = wrap_value((value - val_min) / (val_max - val_min), wrap)*scale

    # only use RGB, not RGBA
    color_code = color_func(value)[:-1]
    
    color_code = ', '.join(["{:0.2f}".format(x) for x in color_code])
    return r"\cellcolor[rgb]{" + color_code + "}"

def is_valid_float(val):
    try:
        float(val)
    except ValueError:
        return False
    else:
        if val != np.inf and val == val:
            return True
        else:
            return False

def return_valid_number_idx(data_array):
    """return the index of data ceil that has valid nummerical value
    """
    is_numeric_3 = np.vectorize(is_valid_float, otypes = [bool])
    return is_numeric_3(data_array)

    
def print_table(data_array, column_tag, row_tag, 
                print_format = "1.2f", 
                with_color_cell = True,
                colormap='Greys', 
                colorscale = 0.5, 
                colorwrap = 0, 
                col_sep = '', 
                print_latex_table=True, 
                print_text_table=True,
                print_format_along_row=True,
                color_minmax_in = 'global',
                pad_data_column = 0,
                pad_dummy_col = 0):
    """
    print a latex table given the data (np.array) and tags    
    step1. table will be normalized so that values will be (0, 1.0)
    step2. each normalzied_table[i,j] will be assigned a RGB color tuple 
           based on color_func( normalzied_table[i,j] * color_scale)
    input
    -----
      data_array: np.array [M, N]
      column_tag: list of str, length N, tag in the first row
      row_tag: list of str, length M, tags in first col of each row
      
      print_format: str or list of str, specify the format to print number
                    default "1.2f"
      print_format_along_row: bool, when print_format is a list, is this
                    list specified for rows? Default True
                    If True, row[n] will use print_format[n]
                    If False, col[n] will use print_format[n]
      with_color_cell: bool, default True,
                      whether to use color in each latex cell
      colormap: str, color map name (matplotlib)
      colorscale: float, default 0.5, 
                    normalized table value will be scaled 
                    color = color_func(nomrlized_table[i,j] * colorscale)
                  list of float
                    depends on configuration of color_minmax_in
                    if color_minmax_in = 'row', colorscale[i] for the i-th row
                    if color_minmax_in = 'col', colorscale[j] for the j-th row
                  np.array
                    color_minmax_in cannot be 'row' or 'col'. 
                    colorscale[i, j] is used for normalized_table[i, j]
      colorwrap: float, default 0, wrap the color-value mapping curve
                 colorwrap > 0 works like mels-scale curve
      col_sep: str, additional string to separate columns. 
               You may use '\t' or ',' for CSV
      print_latex_table: bool, print the table as latex command (default True)
      print_text_table: bool, print the table as text format (default True)
      color_minmax_in: how to decide the max and min to compute cell color?
                 'global': get the max and min values from the input matrix 
                 'row': get the max and min values from the current row
                 'col': get the max and min values from the current column
                  (min, max): given the min and max values
                 default is global
      pad_data_column: int, pad columns on the left or right of data matrix
                  (the tag column will still be on the left)
                  0: no padding (default)
                  -N: pad N dummy data columns to the left
                   N: pad N dummy data columns to the right
      pad_dummy_col: int, pad columns to the left or right of the table
                  (the column will be padded to the left of head column)
                  0: no padding (default)
                  N: pad N columns to the left
    output
    ------
      latext_table, text_table
      
    Tables will be printed to the screen.
    The latex table will be surrounded by begin{tabular}...end{tabular}
    It can be directly pasted to latex file.
    However, it requires usepackage{colortbl} to show color in table cell.    
    """
    if column_tag is None:
        column_tag = ["" for data in data_array[0, :]]
    if row_tag is None:
        row_tag = ["" for data in data_array]

    if pad_data_column < 0:
        column_tag = ["" for x in range(-pad_data_column)] + column_tag
        dummy_col = np.zeros([data_array.shape[0], -pad_data_column]) + np.nan
        data_array = np.concatenate([dummy_col, data_array], axis=1)
    elif pad_data_column > 0:
        column_tag = ["" for x in range(pad_data_column)] + column_tag
        dummy_col = np.zeros([data_array.shape[0], pad_data_column]) + np.nan
        data_array = np.concatenate([data_array, dummy_col], axis=1)
    else:
        pass

    # check print_format
    if type(print_format) is not list:
        if print_format_along_row:
            # repeat the tag
            print_format = [print_format for x in row_tag]
        else:
            print_format = [print_format for x in column_tag]
    else:
        if print_format_along_row:
            assert len(print_format) == len(row_tag)
        else:
            assert len(print_format) == len(column_tag)


    # color configuration
    color_func = cm.get_cmap(colormap)
    #data_idx = return_valid_number_idx(data_array)    
    #value_min = np.min(data_array[data_idx])
    #value_max = np.max(data_array[data_idx])
    
    def get_latex_color(data_array, row_idx, col_idx, color_minmax_in):
        x = data_array[row_idx, col_idx]
        if color_minmax_in == 'row':
            data_idx = return_valid_number_idx(data_array[row_idx])
            value_min = np.min(data_array[row_idx][data_idx])
            value_max = np.max(data_array[row_idx][data_idx])
            if type(colorscale) is list:
                colorscale_tmp = colorscale[row_idx]
        elif color_minmax_in == 'col':
            data_idx = return_valid_number_idx(data_array[:, col_idx])
            value_min = np.min(data_array[:, col_idx][data_idx])
            value_max = np.max(data_array[:, col_idx][data_idx])    
            if type(colorscale) is list:
                colorscale_tmp = colorscale[col_idx]
        elif type(color_minmax_in) is tuple or type(color_minmax_in) is list:
            value_min = color_minmax_in[0]
            value_max = color_minmax_in[1]
            if type(colorscale) is np.ndarray:
                colorscale_tmp = colorscale[row_idx, col_idx]
        else:
            data_idx = return_valid_number_idx(data_array)
            value_min = np.min(data_array[data_idx])
            value_max = np.max(data_array[data_idx])
            if type(colorscale) is np.ndarray:
                colorscale_tmp = colorscale[row_idx, col_idx]
            
        if type(colorscale) is not list:
            colorscale_tmp = colorscale
            

        # return a color command for latex cell
        return return_latex_color_cell(x, value_min, value_max, 
                                       colorscale_tmp, colorwrap, color_func)
    
    # maximum width for tags in 1st column
    row_tag_max_len = max([len(x) for x in row_tag])

    # maximum width for data and tags for other columns
    if print_format_along_row:
        tmp_len = []
        for idx, data_row in enumerate(data_array):
            tmp_len.append(
                max([len("{num:{form}}".format(num=x, form=print_format[idx])) \
                     for x in data_row]))
    else:
        tmp_len = []
        for idx, data_col in enumerate(data_array.T):
            tmp_len.append(
                max([len("{num:{form}}".format(num=x, form=print_format[idx])) \
                     for x in data_col]))
    col_tag_max_len = max([len(x) for x in column_tag] + tmp_len)
    
    # prepare buffer
    text_buffer = ""
    latex_buffer = ""
    text_cell_buffer = []
    latex_cell_buffer = []

    # latex head
    if pad_dummy_col > 0:
        latex_buffer += r"\begin{tabular}{" \
                        + ''.join(['c' for x in column_tag + ['']])
        latex_buffer += ''.join(['c' for x in range(pad_dummy_col)]) + r"}"+"\n"
    else:
        latex_buffer += r"\begin{tabular}{" \
                        + ''.join(['c' for x in column_tag + ['']]) + r"}"+"\n"

    latex_buffer += r"\toprule" + "\n"
    
    # head row
    #  for latex
    hrow = [fill_cell("", row_tag_max_len)] \
           + [fill_cell(x, col_tag_max_len) for x in column_tag]
    if pad_dummy_col > 0:
        hrow = [fill_cell("", 1) for x in range(pad_dummy_col)] + hrow

    latex_buffer += return_one_row_latex(hrow)
    latex_buffer += r"\midrule" + "\n"

    latex_cell_buffer.append(hrow)

    #  for plain text (add additional separator for each column)
    hrow = [fill_cell("", row_tag_max_len, col_sep)] \
           + [fill_cell(x, col_tag_max_len, col_sep) for x in column_tag]
    text_buffer += return_one_row_text(hrow)
    text_cell_buffer.append(hrow)

    # contents
    row = data_array.shape[0]
    col = data_array.shape[1]
    for row_idx in np.arange(row):
        # row head
        row_content_latex = [fill_cell(row_tag[row_idx], row_tag_max_len)]
        row_content_text = [fill_cell(row_tag[row_idx],row_tag_max_len,col_sep)]
        
        if pad_dummy_col > 0:
            row_content_latex = [fill_cell("", 1) for x in range(pad_dummy_col)] \
                                + row_content_latex

        # each column in the raw
        for col_idx in np.arange(col):

            if print_format_along_row:
                tmp_print_format = print_format[row_idx]
            else:
                tmp_print_format = print_format[col_idx]

            if is_valid_float(data_array[row_idx,col_idx]):
                num_str = "{num:{form}}".format(num=data_array[row_idx,col_idx],
                                                form=tmp_print_format)
                latex_color_cell = get_latex_color(data_array, row_idx, col_idx,
                                                   color_minmax_in)
            elif type(data_array[row_idx,col_idx]) is str:
                num_str = "{num:{form}}".format(num=data_array[row_idx,col_idx],
                                                form=tmp_print_format)
                latex_color_cell = ''
            else:
                num_str = ''
                latex_color_cell = ''
                
            if not with_color_cell:
                latex_color_cell = ''
                
            row_content_text.append(
                fill_cell(num_str, col_tag_max_len, col_sep))

            row_content_latex.append(
                fill_cell(latex_color_cell + ' ' + num_str, col_tag_max_len))
            
        # latex table content
        latex_buffer += return_one_row_latex(row_content_latex)
        latex_cell_buffer.append(row_content_latex)
        # text content
        text_buffer += return_one_row_text(row_content_text)
        text_cell_buffer.append(row_content_text)

    latex_buffer += r"\bottomrule" + "\n"
    latex_buffer += r"\end{tabular}" + "\n"

    if print_latex_table:
        print(latex_buffer)
    if print_text_table:
        print(text_buffer)
    return latex_buffer, text_buffer, latex_cell_buffer, text_cell_buffer

class ConfigLA:
    """Configuration to load and parse LA track protocol and score file
    """
    def __init__(self):

        self.pooled_tag = 'Pooled'        
        self.subset_col = 'subset'
        self.score_col = 'score'
        self.index_col = 'trial'
                
        # =====
        # Configuration to load CM protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names = ['spk', self.index_col, 'codec', 'trans', 
                        'attack', 'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names = [self.index_col, self.score_col]
        
        # CM protocol path
        self.protocol_cm_file = 'LA/CM/trial_metadata.txt'


        # =====
        # Configuration to load ASV protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names_asv = ['spk', self.index_col, 'codec', 'trans', 
                            'attack', 'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names_asv = ['asv_spk', self.index_col, self.score_col]

        # ASV protocol path
        self.protocol_asv_file = 'LA/ASV/trial_metadata.txt'
        # ASV score by organizers
        self.pre_score_asv_file = 'LA/ASV/ASVTorch_Kaldi/score.txt'
        # flag, whether tDCF is applicable to this track 
        self.flag_tDCF = True
        
        
        # =====
        # C012 for tDCF computation
        # =====
        # C012 buffer
        self.c012_file = {'eval': 'LA/LA-C012-eval.npy',
                          'progress': 'LA/LA-C012-prog.npy',
                          'hidden': 'LA/LA-C012-hidden.npy'}

        # =====
        # Factors over which the EERs and min t-DCF values are computed
        # =====
        # 1st group of factor
        # name of the data series in protocol dataframe
        self.factor_name_1 = 'attack'
        # value of the factor to be considered
        self.factor_1_list =  ['A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 
                               'A14', 'A15', 'A16', 'A17', 'A18', 'A19', self.pooled_tag]
        # type of the factor (spoofed only? bonafide only? or both)
        self.factor_1_type = g_factor_type_spoof
        # string of factors to be printed
        self.factor_1_tag_list = self.factor_1_list

        # 2nd group of factor
        self.factor_name_2 = 'codec'
        self.factor_2_list = ['none', 'alaw', 'pstn', 'g722', 'ulaw', 'gsm', 'opus', self.pooled_tag]
        self.factor_2_type = g_factor_type_both
        self.factor_2_tag_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', self.pooled_tag]
        return


class ConfigPA:
    """Configuration to load and parse PA track protocol and score file
    """
    def __init__(self):

        self.index_col = 'trial'
        self.subset_col = 'subset'
        self.pooled_tag = 'Pooled'
        self.score_col = 'score'

        # =====
        # Configuration to load CM protocol and score file
        # =====
        # PA_0010 PA_E_1000001 R3 M3 d4 r1 m1 s4 c4 spoof notrim eval
        # name of data series for procotol file
        self.p_names = ['spk', self.index_col, 'asv_room', 'asv_mic', 'dis_to_asv', 
                        'att_room', 'att_mic', 'att_d', 'att_to_spk', 
                        'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names = [self.index_col, self.score_col]

        # CM protocol path
        self.protocol_cm_file = 'PA/CM/trial_metadata.txt'


        # =====
        # Configuration to load ASV protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names_asv = ['spk', self.index_col, 'asv_room', 'asv_mic', 'dis_to_asv', 
                            'att_room', 'att_mic', 'att_d', 'att_to_spk', 
                            'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names_asv = ['asv_spk', self.index_col, self.score_col]

        # ASV protocol path
        self.protocol_asv_file = 'PA/ASV/trial_metadata.txt'
        # ASV score by organizers
        self.pre_score_asv_file = 'PA/ASV/ASVTorch_Kaldi/score.txt'
        # flag, whether tDCF is applicable to this track 
        self.flag_tDCF = True


        # =====
        # special for PA hidden track, we have two
        # =====        
        # hidden subset 1
        self.hidden = {'hidden1_PA': 'trim == "notrim" and subset == "hidden"',
                       'hidden2_PA': 'trim == "trim" and subset == "hidden"'}

        # =====
        # C012 for tDCF computation
        # =====
        # C012 buffer
        self.c012_file = {'eval': 'PA/PA-C012-eval.npy',
                          'progress': 'PA/PA-C012-prog.npy',
                          'hidden1_PA': 'PA/PA-C012-hidden1.npy',
                          'hidden2_PA': 'PA/PA-C012-hidden2.npy'}


        # =====
        # Factors over which the EERs and min t-DCF values are computed
        # =====        
        # 1st group of factor
        # name of the data series in protocol dataframe
        # we will concatenate multiple factors into one group
        # dummy is a placeholder where we store the value for pooled condition
        self.factor_name_1 = ['asv_room', 'asv_mic', 'dis_to_asv', 'dummy']
        # value of the factor to be considered
        self.factor_1_list =  [['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'], 
                               ['M1', 'M2', 'M3'], 
                               ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'], 
                               [self.pooled_tag]]
        # type of the factor (spoofed only? bonafide only? or both)
        self.factor_1_type = [g_factor_type_both, g_factor_type_both, 
                              g_factor_type_bonafide,  g_factor_type_both]
        # string of factors to be printed
        self.factor_1_tag_list = [item for sublist in self.factor_1_list for item in sublist]


        # 2nd group of factor
        self.factor_name_2 = ['att_room', 'att_mic', 'att_to_spk', 'att_d', 'dis_to_asv', 'dummy']
        self.factor_2_list = [['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'],
                              ['m1', 'm2', 'm3'],
                              ['c2', 'c3', 'c4'],
                              ['s2', 's3', 's4'],
                              ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'],
                              [self.pooled_tag]]
        self.factor_2_type = [g_factor_type_spoof, g_factor_type_spoof,
                              g_factor_type_spoof, g_factor_type_spoof,
                              g_factor_type_spoof, g_factor_type_both]
        self.factor_2_tag_list = [item for sublist in self.factor_2_list for item in sublist]

        return

class ConfigDF:
    """Configuration to load and parse LA track protocol and score file
    """
    def __init__(self):
       
        self.index_col = 'trial'
        self.subset_col = 'subset'
        self.pooled_tag = 'Pooled'
        self.score_col = 'score'

        # =====
        # Configuration to load CM protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names = ['speaker', self.index_col, 'compr', 'source', 'attack',
                        'label', 'trim', 'subset', 'vocoder', 
                        'task', 'team', 'gender-pair', 'language']
        # name of data series for score file
        self.s_names = [self.index_col, self.score_col]
        
        # Path to the CM protocol file
        self.protocol_cm_file = 'DF/CM/trial_metadata.txt'
        
        # =====
        # Configuration to load ASV protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names_asv = []
        self.s_names_asv = []
        # ASV protocol
        self.protocol_asv_file = ''
        self.pre_score_asv_file = ''
        # flag, whether tDCF is applicable to this track 
        self.flag_tDCF = False

        # =====
        # C012 for tDCF computation
        # =====
        # C012 buffer
        self.c012_file = {'eval': '',
                          'progress': '',
                          'hidden': ''}

        
        # =====
        # Factors over which the EERs and min t-DCF values are computed
        # =====
        # 1st group of factor
        # name of the data series in protocol dataframe
        self.factor_name_1 = 'vocoder'
        self.factor_1_list =  ['traditional_vocoder', 
                               'waveform_concatenation', 
                               'neural_vocoder_autoregressive', 
                               'neural_vocoder_nonautoregressive', 
                               'unknown', self.pooled_tag]
        self.factor_1_type = g_factor_type_spoof
        self.factor_1_tag_list = ['Traditional', 'Wav.Concat.', 'Neural AR', 
                                  'Neural non-AR', 'Unknown', self.pooled_tag]

        self.factor_name_2 = 'compr'
        self.factor_2_list = ['nocodec',  'low_mp3', 'high_mp3', 'low_m4a', 
                              'high_m4a', 'low_ogg', 'high_ogg', 'mp3m4a', 
                              'oggm4a', self.pooled_tag]
        self.factor_2_type = g_factor_type_both
        self.factor_2_tag_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
                                  'C7', 'C8', 'C9', self.pooled_tag]
        
        return



def evaluation_API(cm_score_file, track, subset = 'eval', label_dir = './',
                   flag_recompute_c012 = False,  
                   asv_score_file = None, 
                   external_c012_path = None):
  
    """ mintdcf_array, eer_array = evaluation_API(score_file, track)

    Compute the min tDCF and EER values given a score file.
    The output shares the same format as that on CodaLab page.

    input
    -----
      cm_score_file   str, path to the CM score file
      track           str, 'LA', 'PA', or 'DF'
      subset          str, name of subset, eval, progress, or hidden
      label_dir       str, path to the directory of key and meta labels
                      label_dir is the directory downloaded from ASVspoof.org
                      It should contain the following files
                      \- DF
                         \- CM 
                             |- trial_metadata.txt
                             |- ...
                      \- LA
                         ...
                      \- PA
                         ...
      flag_recompute_c012 bool, whether recompute C012 coef                
                          default False

      asv_score_file      str, path to the ASV score file, default None    
                          If None, ASV score in label_dir will be loaded   
                                                                     
                                                                          
      external_c012_path  str, path to external C012 file                  
                          If None, path is specified by config.py          
                                                                          
    output
    ------
      mintdcf_array   np.array, the numpy array of min t-DCF
      eer_array       np.array, the numpy array of EER
    """
    
    # ===========
    # load configuration for each trakc
    # ===========
    if track_name == g_LA_track:
        config_buf = ConfigLA()
    elif track_name == g_PA_track:
        config_buf = ConfigPA()
    elif track_name == g_DF_track:
        config_buf = ConfigDF()
    else:
        print("ERROR: unknown track: {:s}".format(track_name))
        return None

    # ===========
    # load CM protocol & score
    # ===========
    protocol_cm_file = os.path.join(label_dir, config_buf.protocol_cm_file)
    protocol_cm_pd = load_protocol(protocol_cm_file, 
                                   names = config_buf.p_names, 
                                   index_col = config_buf.index_col)
    # load score file
    score_cm_pd = load_score(cm_score_file, config_buf.s_names, 
                             index_col = config_buf.index_col)
    # merge score and protocol into a single dataFrame
    score_cm_pd = join_protocol_score(
        protocol_cm_pd, score_cm_pd[[config_buf.score_col]])

    # ===========
    # select the subset
    # ===========
    if track_name == g_PA_track and subset in config_buf.hidden:
        # special for PA
        subset_query = config_buf.hidden[subset]
    else:
        # other cases
        subset_query = '{:s} == "{:s}"'.format(config_buf.subset_col, subset)
    # get the evaluation subset data frame
    tmp_score_cm_pd = score_cm_pd.query(subset_query)


    # ===========
    # on C012
    # ===========
    
    # specify the path C012 dictionary
    if external_c012_path is None or len(external_c012_path) == 0:
        # use pre-computed C012
        c012_file = os.path.join(label_dir, config_buf.c012_file[subset])
    else:
        c012_file = external_c012_path
        
        
    # compute C012 if necessary
    if config_buf.flag_tDCF and flag_recompute_c012:
        # protocol ASV
        protocol_asv_file = os.path.join(label_dir, config_buf.protocol_asv_file)
        if not os.path.isfile(protocol_asv_file):
            print("Cannot find ASV protocol {:s}".format(protocol_asv_file))
            sys.exit(1)
        # score ASV
        if asv_score_file is None or len(asv_score_file) == 0:
            # use pre-computed ASV score
            asv_score_file = os.path.join(label_dir, config_buf.pre_score_asv_file)
        if not os.path.isfile(asv_score_file):
            print("Cannot find ASV score file {:s}".format(asv_score_file))
            sys.exit(1)

        print("===============\nCompute C012 coef\n===============")
        protocol_asv_pd = load_protocol(protocol_asv_file, 
                                        names = config_buf.p_names_asv)
        # load score file
        asv_score_pd = load_score(
            asv_score_file, config_buf.s_names_asv)
        # merge score and protocol into a single dataFrame
        asv_score_pd = join_protocol_score(
            protocol_asv_pd, asv_score_pd[[config_buf.score_col]])

        # get the evaluation subset data frame
        tmp_asv_score_pd = asv_score_pd.query(subset_query)
            
        C012_buf = compute_tDCF_C012(tmp_asv_score_pd, 
                                     config_buf.factor_name_1, 
                                     config_buf.factor_1_list, 
                                     config_buf.factor_1_type,
                                     config_buf.factor_name_2, 
                                     config_buf.factor_2_list, 
                                     config_buf.factor_2_type)
        dump_C012_dict(C012_buf, c012_file)
        print("Save C012 coef to {:s}".format(c012_file))
            
    
    # load C012 dictionary
    if config_buf.flag_tDCF:
        if not os.path.isfile(c012_file):
            print("Cannot find C012 file {:s}".format(c012_file))
            sys.exit(1)
        print("=============== \nCompute EERs, min tDCFs\n===============")
        print("Load C012 coeffs from {:s}".format(c012_file))
        C012_buf = load_C012_dict(c012_file)        
    else:
        print("========== \nCompute EERs\n==========")
        print("Track without considering ASV")
        C012_buf = None
    
    # ===========        
    # compute min tDCF and EERs
    # ===========
    mintdcf_array, eer_array, thres_array = compute_decomposed_mintdcf_eer(
        tmp_score_cm_pd, 
        config_buf.factor_name_1, 
        config_buf.factor_1_list, 
        config_buf.factor_1_type,
        config_buf.factor_name_2, 
        config_buf.factor_2_list, 
        config_buf.factor_2_type,
        C012_buf = C012_buf,
        pooled_tag = config_buf.pooled_tag, 
        col_score_name = config_buf.score_col,
        flag_verbose = False)
    
    # ===========
    # print results
    # ===========
    print("\n\n")
    # print min tDCF table
    if C012_buf is not None:
        print("\n===============\nTable for min tDCFs\n===============\n")
        print_table(mintdcf_array, 
                    config_buf.factor_2_tag_list, 
                    config_buf.factor_1_tag_list, 
                    print_format = "1.4f", 
                    with_color_cell = True,
                    print_latex_table=True, 
                    print_text_table=True);

    # print EER table
    print("\n===============\nTable for EERs\n===============\n")
    print_table(eer_array * 100, 
                config_buf.factor_2_tag_list, 
                config_buf.factor_1_tag_list, 
                print_format = "1.2f", 
                with_color_cell = True,
                print_latex_table=False, 
                print_text_table=True);

    # print Threshold table
    print("\n===============\nTable for Threshold\n===============\n")
    print_table(thres_array, 
                config_buf.factor_2_tag_list, 
                config_buf.factor_1_tag_list, 
                print_format = "1.6f", 
                with_color_cell = True,
                print_latex_table=False, 
                print_text_table=True);

    return mintdcf_array, eer_array, thres_array


def clean_package(prjdir, package_name, score_file_name):
    # delete existing file
    if os.path.isfile(os.path.join(prjdir, package_name)):
        os.system("rm {:s}".format(os.path.join(prjdir, package_name)))
    if os.path.isfile(os.path.join(prjdir, score_file_name)):
        os.system("rm {:s}".format(os.path.join(prjdir, score_file_name)))
    return

def untar_package(prjdir, package_name, score_file_name):
    if package_name.endswith('.zip'):
        os.system('unzip ' + os.path.join(prjdir, package_name))
    elif package_name.endswith('.tar.gz'):
        os.system('tar -xzf ' + os.path.join(prjdir, package_name))
    elif package_name.endswith('.txt'):
        print("uploading txt file may be slow. You may upload .zip or .tar.gz")
    else:
        print("Please upload .zip, or .tar.gz")

    if not os.path.isfile(os.path.join(prjdir, score_file_name)):
        print("Cannot find the score file: {:s}".format(score_file_name))
    else:
        print("We will use {:s}".format(os.path.join(prjdir, score_file_name)))
    return



# Pointer to data directory on Google Colab runtime
prjdir = ''

track_name = 'DF' #@param ["LA", "PA", "DF"]
subset_name = 'eval' #@param ["eval", "progress", "hidden", 'hidden1_PA', 'hidden2_PA']

score_file_name = sys.argv[1]

flag_compute_c012 = False #@param ["False", "True"] {type:"raw"}
score_file_asv_name = 'asv_score.txt' #@param {type:"string"}
package_asv_name = 'asv_score.txt.zip' #@param {type:"string"}
external_c012_path = 'c012.npy' #@param {type:"string"}

print("Compute result using {:s}, track: {:s}, subset: {:s}".format(score_file_name, track_name, subset_name))

if not flag_compute_c012:
    # when external_c012_path is None, the code will load pre-computed C012
    external_c012_path = None

min_tdcfs, eers, thress = evaluation_API(prjdir + score_file_name, 
                                 track = track_name, 
                                 subset = subset_name, 
                                 label_dir = os.path.join(prjdir, 'keys'),
                                 flag_recompute_c012 = flag_compute_c012,
                                 asv_score_file = os.path.join(prjdir, score_file_asv_name),
                                 external_c012_path = external_c012_path)


