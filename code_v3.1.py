import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from itertools import product
import copy
import time

class galaxy:
    def __init__(self, ypos, xpos, radius, count_sum, bg_galaxy=3419):  ### xdim, ydim??? and Angle
        self.ypos = ypos
        self.xpos = xpos
        self.radius = radius
        self.count_sum = count_sum
        self.bg_galaxy = bg_galaxy
        # Maybe some loss function to calculate counting

    def get_info(self):
        """ Return x, y position, radius, sum of pixels and local background"""
        return np.array([self.ypos, self.xpos, self.radius, self.count_sum, self.bg_galaxy])
    
    

class analysis:
    def __init__(self, fname):
        self.fname = fname
        self.raw_image_data = fits.getdata(self.fname, ext=0)[3820:3975,422:570]
        self.shapes = [self.raw_image_data.shape[0], self.raw_image_data.shape[1]]
        self.dimension = self.shapes[0] * self.shapes[1]
        # creating masked array
        self.masked = np.zeros(self.shapes)
        for j, i in product(np.arange(0, self.shapes[0]), np.arange(0, self.shapes[1])):
            self.masked[j, i] = 1
        # ranking the image_data from the highest counts to the lowest
        self.image_data = copy.deepcopy(self.raw_image_data)  # Creating a deep copy of the original data, not advisable for a large data file
        self.image_data = self.image_data.reshape([self.shapes[0] * self.shapes[1]])  # ranked image
        self.rank = np.argsort(self.image_data)[::-1]  # the corresponding position of the
        self.image_data.sort(axis=0)
        self.image_data = self.image_data[::-1]
        self.galaxies = []
        print(self.raw_image_data[116,79])


    def rank_yx(self, rankyx, rank_to_yx=1):
        """ Transform rank (single value) to x,y position if rank_to_yx = 1 (True)"""
        if rank_to_yx == 1:
            x = int(rankyx) % int(self.shapes[1])
            y = (rankyx - x) / int(self.shapes[1])
            return [y, x]  # More convenient to return y, x
        
        if rank_to_yx == 0:  # that means transfer yx to rank, expecting rankyx to be a list
            rankyx = rankyx[0] * int(self.shapes[1]) + rankyx[1]
            return rankyx  # returns back a float

    def mask_region(self, ypos, xpos, r):
        """ masking the circular region at radius r"""
        for j, i in product(np.arange(ypos - r, ypos + r + 1), np.arange(xpos - r, xpos + 1 + r)):  # Create square
            if (j - ypos) ** 2 + (i - xpos) ** 2 <= r ** 2 and 0 <= j<= self.shapes[0] - 1 and 0<= i <=self.shapes[1] - 1:
                j = int(j)
                i = int(i)
                # print(j,i)
                self.masked[j, i] = 0

    def pick_largest(self, cut_off):
        """ Look for the largest unmasked value"""
        for i in range(self.dimension):
            # print(self.rank[i],"corr rank", self.rank_yx(self.rank[i]))
            m = self.masked[int(self.rank_yx(self.rank[i])[0])
                            ,int(self.rank_yx(self.rank[i])[1])]
            # print(i, self.image_data[i], m)
            if m * self.image_data[i] == self.image_data[i]:
                # self.masked.reshape([self.shapes[0], self.shapes[1]])
                if self.image_data[i] <= cut_off:
                    print("Surveying completed")
                    return -1,-1  # returns none if scan is completed
                else:
                    # print("count and rank =", self.image_data[i], np.array(self.rank[i]))
                    return self.image_data[i], np.array(self.rank[i])

    def fit_galaxy(self, ypos, xpos, r_in, r_out = 0):
        """ fit the galaxy to a circle of radius r """
        count_out = []
        count_in = []
        for j, i in product(np.arange(ypos - (r_out + r_in), ypos + r_out + r_in + 1),np.arange(xpos - (r_out + r_in), xpos + 1 + r_out + r_in)):  # Create square
            if (j - ypos) ** 2 + (i - xpos) ** 2 <= r_in ** 2 and 0<= j <= self.shapes[0] - 1 and 0<= i <= self.shapes[1] - 1: # make sure points are in a circle
                j = int(j)
                i = int(i)
                if self.raw_image_data[j,i] * self.masked[j,i] == self.raw_image_data[j,i]:
                    count_in.append(self.raw_image_data[j,i])
            if r_in ** 2 < (j - ypos) ** 2 + (i - xpos) ** 2 <= (r_in + r_out)**2 and 0<= j <= (self.shapes[0] - 1) and 0<= i  <= self.shapes[1] - 1: # in the outer ring
                j = int(j)
                i = int(i)
                if self.raw_image_data[j,i] * self.masked[j,i] == self.raw_image_data[j,i]:        
                    count_out.append(self.raw_image_data[j][i]) 
        return count_in, count_out



    def scan_ap(self, cut_off = 3480, r_ap = 12, r_an = 3):
        """ Do the aperture method scan by setting radius for the aperture and the annulus"""
        for trial in range(self.dimension):
          
            max_count, max_rank = self.pick_largest(cut_off = cut_off)
            if max_count >= 0:
                y,x = self.rank_yx(max_rank)
                print("Scan  pos", y,x," scanning",trial,"counts", max_count)
                count_in, count_out = self.fit_galaxy(y,x,r_ap, r_an)
                count_sum = []
                local_bg = []
                for c in range(len(count_in)):
                    if count_in[c] >= cut_off:
                        count_sum.append(count_in[c])
                for c in range(len(count_out)):
                    if count_out[c] <= cut_off:
                        local_bg.append(count_out[c])
                
                if len(count_sum) >= int(np.pi * (r_ap **2) / 2):  # Make sure it is not noise
                    count_sum = np.array(count_sum).sum()
                    if len(local_bg) != 0:
                        total = 0
                        for c in range(len(local_bg)):
                            if 3*13.8 <= abs(local_bg[c] - 3419):
                                total += local_bg[c]
                        local_bg = total / len(local_bg)
                    else:
                        local_bg = 3419
                    print("galaxy founded at ", y, x)
                    self.galaxies.append(galaxy(y, x, r_ap, count_sum, bg_galaxy=local_bg))
                # self.masked.reshape([self.shapes[0], self.shapes[1]])
           
            elif max_count == -1:
                print("aperture scan completed, number of galaxies found is", len(self.galaxies))
                break
            self.mask_region(y, x, r_ap+r_an)

###########################################################################################################
    def scan(self, cut_off):
        """ A variation on the aperture method.The idea is to
        start at radius 2 and increase the radius to get a better fit on the galaxy"""
    
                         # list of pixels within the region "galaxy"
        start_time = time.time()
        for trial in range(self.dimension):
            max_count, max_rank = self.pick_largest(cut_off = cut_off)
            fitted = 0                   # galay not fitted to a circle
            point = []  
            
            if max_count == -1:
                print("Scan completed, number of galaxies found is ", len(self.galaxies))
                break
            
            if max_count >= 0:
                ypos,xpos = self.rank_yx(max_rank)
                # print("max_count, y, x", max_count, ypos, xpos)    
            
            
            for r in range(4, 80, 4):     # r = radius, we know the largest radius can't be >500 as it will be 1/4 of the pic.
                # print("locating the galaxy position at", ypos, xpos, "at a radius r =", r)               
                if fitted == 1 or r == 80:
                    # print("max_count, yx",max_count, ypos, xpos, "cut =", no_cut, len(new_point)/2)
                    self.mask_region(ypos, xpos, r - 2)
                    if r - 4 >= 4:               # Too small so assume counting error
                        if no_bg > 3:
                            bg = bg_local / no_bg
                        else:
                            bg = 3419
                        self.galaxies.append(galaxy(ypos, xpos, r -4, np.array(point).sum(), bg))
                        print("\nscan of galaxy completed at radius", r - 4, "position =", ypos, xpos)
                    # print("run time = ", time.time() - start_time)
                    break
                
                ##### Resetting parameters ####
                no_bg = 0              # number 0f background pixels
                bg_local = 0           # sum of local background noises
                no_cut = 0
                new_point = []         # pending pixels to be added
                ###############################
                
                if fitted == 0:
                    
                    if r == 4:
                        for j, i in product(np.arange(ypos - r, ypos + r + 1), np.arange(xpos - r, xpos + 1 + r)):  # Create square
            
                            # Check if it is within the circle radius = 2
                            if int((i - xpos) ** 2 + (j - ypos) ** 2) <= r ** 2 and 0<= j <= (self.shapes[0] - 1) and 0<= i  <= self.shapes[1] - 1:
                                i,j =[int(i), int(j)]                  
                                if self.raw_image_data[j,i] == self.raw_image_data[j,i] *self.masked[j,i]:    # Append the ppoint if not masked (masked = 1)
                                    point.append(self.raw_image_data[j,i])
                                    
                                    if self.raw_image_data[j,i] <= cut_off:
                                        no_cut += 1
                                    if abs(self.raw_image_data[j,i] -3419) <= 3.5*13.8:
                                        bg_local += self.raw_image_data[j,i]
                                        no_bg += 1
                        if no_cut > len(point)/2:
                            fitted = 1
                        else:
                            pass
                        
                        print("max_count, yx",max_count, ypos, xpos, "cut =", no_cut, len(point)/2)
                       
#########################################################
                    if r > 4:
                        
                        # print("largest pixel =", image_data[0:5], "Corresponding rank", rank[0:5])
                        for j, i in product(np.arange(ypos - r, ypos + r + 1), np.arange(xpos - r, xpos + 1 + r)):              
                            # Check if data are in between the previous and the new circle
                            if (r - 4)**2 < int((i - xpos) ** 2 + (j - ypos) ** 2) <= r ** 2 and 0<= j <= (self.shapes[0] - 1) and 0<= i  <= self.shapes[1] - 1:
                                i,j =[int(i), int(j)]               # just incase    
                                
                                if self.raw_image_data[j,i] * self.masked[j,i] == self.raw_image_data[j,i]:
                                    new_point.append(self.raw_image_data[j, i])   # points are pending to be added in
                                    if self.raw_image_data[j,i] <= cut_off:
                                        no_cut += 1
                                    if abs(self.raw_image_data[j,i] -3419) <= 3.5*13.8:
                                        bg_local += self.raw_image_data[j,i]
                                        no_bg += 1
                        
                        # Check if half of the new data points are inside cut off region
                        if no_cut <= int(len(new_point))/2:
                            for rannk in range(len(new_point)):
                                point.append(new_point[rannk])
                        
                        else:
                            fitted = 1
                    
                        print("max_count, yx",max_count, ypos, xpos, "cut =", no_cut, len(new_point)/2)


ana = analysis("A1_pic_bleed_filtered.fits")
ana.scan(4000)

