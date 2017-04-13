import fpf_calculator
import glob, os
import numpy as np
import operator

folder_list = ["mirr_soft/",
               "zero_hard/",
               "zero_soft/",
               "no/",]

# planet_pos = (65, 63) # BP 29
planet_pos = (58.5, 67.5)  # HR8799 e
# planet_pos = (186.0, 142.0)  # HR8799 e keck
# planet_pos = (47.0, 60.0)  # HR8799 f
# planet_pos = (75, 67) # HR8799 d
#planet_pos = (65, 63) # BP 26
#planet_pos = (27, 77) # HR8799 c

shifts = np.linspace(-2.0, 2.0, num= 20) # HR8799 e

# shifts = np.linspace(-1.0, 1.0, num=5)  # BP 29

#planet_pos = (67, 55) # EPS ERI

planet_shifts = [(planet_pos[0] + x, planet_pos[1] + y) for x in shifts for y in shifts]

print "HR8799"

for tmploc in folder_list:
    print "------------- " + tmploc + "---------------"

    tmp_location = "/Users/markusbonse/Desktop/Results/" + tmploc

    last_planet_pos = planet_pos

    # get all fits files in the directory
    os.chdir(tmp_location)
    for file in glob.glob("*.fits"):

        snr_result_list = []
        planet_pos_list = []

        for shift in planet_shifts:
            tmp_result = fpf_calculator.fpf_calculator(tmp_location + str(file),
                                                       filetype="fits",
                                                       planet_position=shift,
                                                       radius=4.0/2.0,
                                                       method="exact",
                                                       plot=False,
                                                       save=False)

            snr_result_list.append(tmp_result[2])
            planet_pos_list.append(tmp_result[3])

        index, value = max(enumerate(snr_result_list), key=operator.itemgetter(1))

        print str(value)

        '''
        fpf_calculator.fpf_calculator(tmp_location + str(file),
                                      filetype="fits",
                                      planet_position=planet_pos_list[index],
                                      radius=4.0 / 2.0,
                                      method="exact",
                                      plot=True,
                                      save=False)'''


