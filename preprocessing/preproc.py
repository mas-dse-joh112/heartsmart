
methods = ['1','2','3']
paths = ['train','validate','test']
dsbpaths = ['train','validate','test']
sunnybrookpaths = ['challenge_online', 'challenge_validaton']
acdcpaths = ['niftidata']

sources = {"dsb":{"dir":"/opt/data/dsb",
                  "paths":dsbpaths,
                  "string":"study",
                  "pattern":"sax",
                 },
           "sunnybrook":{"dir":"/opt/data/sunnybrook",
                         "paths":sunnybrookpaths,
                         "string":"*",
                         "pattern":"",
                        },
           "acdc":{"dir":"/opt/data/acdc",
                   "paths":acdcpaths,
                   "string":"",
                   "pattern":"",
                  },
          }

normoutputs = {"dsb":{"dir":"/opt/output/dsb/norm",
                 },
           "sunnybrook":{"dir":"/opt/output/sunnybrook/norm",
                        },
           "acdc":{"dir":"/opt/output/acdc/norm",
                  },
          }

method1 = 1
method2 = 0
method3 = 0
method4 = 0


