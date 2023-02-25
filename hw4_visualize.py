import csv
from pyBayes.MCMC_Core import MCMC_Diag

part2 = True
part3 = False
part4 = False

if part2:
    part2_inst = MCMC_Diag()
    part2_MC_sample = []
    with open("hw4_fullbayes_samples_t1.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            part2_MC_sample.append([float(x) for x in row])
    with open("hw4_fullbayes_samples_t2.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            part2_MC_sample.append([float(x) for x in row])
    part2_inst.set_mc_samples_from_list(part2_MC_sample)
    part2_inst.burnin(100)
    #                              0        1        2        3        4        5        6           7        8      9    10          11
    part2_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
    part2_inst.show_traceplot((6,2))
    part2_inst.show_hist((6,2))
    part2_inst.show_acf(30, (6,2))
    part2_inst.print_summaries(4, latex_table_format=True)

if part3:
    part3_inst = MCMC_Diag()
    part3_MC_sample = []
    with open("hw4_bayes_nugget_samples.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            part3_MC_sample.append([float(x) for x in row])
    part3_inst.set_mc_samples_from_list(part3_MC_sample)
    # part3_inst.burnin(100)
    #                                  0        1        2        3        4        5        6           7      8 
    part3_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_S", "phi", "v"])
    part3_inst.show_traceplot((6,2))
    part3_inst.show_hist((6,2))
    part3_inst.show_acf(30, (6,2))
    part3_inst.print_summaries(4, latex_table_format=True)

if part4:
    part4_inst = MCMC_Diag()
    part4_MC_sample = []
    with open("hw4_map_phi_samples.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            part4_MC_sample.append([float(x) for x in row])
    part4_inst.set_mc_samples_from_list(part4_MC_sample)
    #                              0        1        2        3        4        5        6           7        8      9    10          11
    part4_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
    part4_inst.show_traceplot((6,2))
    part4_inst.show_hist((6,2))
    part4_inst.show_acf(30, (6,2))
    part4_inst.print_summaries(4, latex_table_format=True)