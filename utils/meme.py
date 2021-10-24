import os
import numpy as np
from tqdm import tqdm
import re


class motif(object):

    def __init__(self, name,pwm):
        self.name = name
        self.pwm = pwm  
        self.length = pwm.shape[0]        

    def normalize(self):        
        rs = np.sum(self.pwm, axis=1)
        self.pwm = self.pwm / rs.reshape([-1,1])

    def reshape(self,newlength):
        diff = newlength - self.length        
        roll_len = abs(diff)//2 + abs(diff) % 2    
        if self.length > newlength:                            
            self.pwm= np.roll(self.pwm, roll_len, axis=0)
            self.pwm = self.pwm[abs(diff):self.length,:]
        
        if self.length < newlength:                            
            newpwm = np.full((newlength,self.pwm.shape[1]),0.25)
            start = abs(diff)//2
            end = start + self.length
            newpwm[start:end,:] = self.pwm            
            self.pwm= newpwm 
        
        self.length = newlength
        self.normalize()       
        


def __parse_header(handle):

    line1 = handle.readline()
    line2 = handle.readline()
    line3 = handle.readline()
    line4 = handle.readline()
    line5 = handle.readline()
    line6 = handle.readline()
    line7 = handle.readline()
    line8 = handle.readline()
    line9 = handle.readline()

    if not line1.startswith("MEME version"):
        raise ValueError("MEME version should be specified in the 1st line\n")
    
    if line2 != "\n" or line4 != "\n":
        raise ValueError("Header format error. No separator detected\n")
    
    if not line3.startswith("ALPHABET"):
        raise ValueError("No alphabet specified")

    if not line5.startswith("strands:"):
        raise ValueError("No strand specified")
    

    if not line7.startswith("Background letter frequencies"):
        raise ValueError("No bakground frequences are specified (line 5)")
    

def __parseMotif(handle):

    motif_pat = re.compile(r"MOTIF\s(\w+)")    
    name_line = handle.readline()
    match = motif_pat.search(name_line)
    if not match:
        raise ValueError("Motif header format")
    
    name= match.group(1)

    info_line = handle.readline()
    if not info_line.startswith("letter-probability matrix"):
        raise ValueError("No motif info are available for motif %s" % name)

    alength_pat = re.compile(r"alength= (\d+)")
    width_pat = re.compile(r"w= (\d+)")
    
    len_match = alength_pat.search(info_line)
    width_match = width_pat.search(info_line)

    if not len_match:
        raise ValueError("Motif length not specified for motif %s" % name)
    
    if not width_match:
        raise ValueError("Motif width not specified for motif %s" % name)
    
    nb = int(len_match.group(1))
    w = int(width_match.group(1))

    pwm = np.zeros((w,nb))
    for i in range(w):
        line = handle.readline()
        line = line.strip().split()
        if len(line) != nb:
            raise ValueError("The PWM of motif %s has a size different than expected" % name)
        line = [float(x) for x in line]
        pwm[i,:] = line
    
    url = handle.readline()
    line = handle.readline()

    if line != "\n":
        raise ValueError("No separation line at the end of motif %s" % name)

    return motif(name, pwm)

    

    

def parseMeme(fname, max_motifs=1000):
    
    pat_header = re.compile(r"MEME version")
    par_alphabet = re.compile(r"ALPHABET=\s+(\W+)")
    pat_bgrd = re.compile(r"Background letter frequencies")

    # check that the file exists 
    if  not os.path.exists(fname):
        raise FileNotFoundError("Could not find file %s" % fname)

    state =0

    motifs = []
    nloaded = 0
    with open(fname,'r') as fin:
        __parse_header(fin)
        parse = True
        while parse and nloaded < max_motifs:
            try:
                mtf = __parseMotif(fin)
                motifs.append(mtf)
                nloaded +=1
            except:
                parse = False

    return motifs       


            


        
