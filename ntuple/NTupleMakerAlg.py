from AnaAlgorithm.AlgSequence import AlgSequence
from AnaAlgorithm.DualUseConfig import createAlgorithm

def makeSequence () :

  algSeq = AlgSequence()

  # Algorithm for calculating variables to be dumped
  ntupAlg = createAlgorithm( 'NTupleMakerAlg', 'NTupleMakerAlg' )
  algSeq += ntupAlg

  treename = 'sixtop'

  # Add an ntuple dumper algorithm:
  treeMaker = createAlgorithm( 'CP::TreeMakerAlg', 'TreeMaker' )
  treeMaker.TreeName = treename
  algSeq += treeMaker

  ntupleMaker = createAlgorithm( 'CP::AsgxAODNTupleMakerAlg', 'NTupleMakerEventInfo' )
  ntupleMaker.TreeName = treename
  ntupleMaker.Branches = [ 
                            'EventInfo.M_b4_1    -> M_b4_1',
                            'EventInfo.Pt_b4_1    -> Pt_b4_1',
                            'EventInfo.Eta_b4_1    -> Eta_b4_1',
                            'EventInfo.Phi_b4_1    -> Phi_b4_1',
                            'EventInfo.M_b4_2    -> M_b4_2',
                            'EventInfo.Pt_b4_2    -> Pt_b4_2',
                            'EventInfo.Eta_b4_2    -> Eta_b4_2',
                            'EventInfo.Phi_b4_2    -> Phi_b4_2',
                            'EventInfo.M_H1    -> M_H1',
                            'EventInfo.Pt_H1    -> Pt_H1',
                            'EventInfo.Eta_H1    -> Eta_H1',
                            'EventInfo.Phi_H1    -> Phi_H1',
                            'EventInfo.M_H2    -> M_H2',
                            'EventInfo.Pt_H2    -> Pt_H2',
                            'EventInfo.Eta_H2    -> Eta_H2',
                            'EventInfo.Phi_H2    -> Phi_H2',
                           'EventInfo.Pt_bj1     -> Pt_bj1',
                           'EventInfo.Pt_bj2     -> Pt_bj2',
                           'EventInfo.Pt_bj3     -> Pt_bj3',
                           'EventInfo.Pt_bj4     -> Pt_bj4',
                           'EventInfo.Pt_bj5     -> Pt_bj5',
                           'EventInfo.Pt_bj6     -> Pt_bj6',
                           'EventInfo.Eta_bj1    -> Eta_bj1',
                           'EventInfo.Eta_bj2    -> Eta_bj2',
                           'EventInfo.Eta_bj3    -> Eta_bj3',
                           'EventInfo.Eta_bj4    -> Eta_bj4',
                           'EventInfo.Eta_bj5    -> Eta_bj5',
                           'EventInfo.Eta_bj6    -> Eta_bj6',
                           'EventInfo.Phi_bj1    -> Phi_bj1',
                          'EventInfo.Phi_bj2    -> Phi_bj2',
                          'EventInfo.Phi_bj3    -> Phi_bj3',
                          'EventInfo.Phi_bj4    -> Phi_bj4',
                          'EventInfo.Phi_bj5    -> Phi_bj5',
                          'EventInfo.Phi_bj6    -> Phi_bj6',
                           'EventInfo.num_bjets     -> num_bjets',
                           'EventInfo.num_jets     -> num_jets',
                           'EventInfo.HTb     -> HTb',
                           'EventInfo.HT     -> HT',
                           'EventInfo.met_NonInt  -> met_NonInt',
                           'EventInfo.set_NonInt  -> set_NonInt',
                           'EventInfo.Pt_j1     -> Pt_j1',
                           'EventInfo.Pt_j2     -> Pt_j2',
                            'EventInfo.Pt_j3     -> Pt_j3',
                            'EventInfo.Pt_j4     -> Pt_j4',
                            'EventInfo.Pt_j5     -> Pt_j5',
                            'EventInfo.Pt_j6     -> Pt_j6',
                            'EventInfo.Pt_j7     -> Pt_j7',
                            'EventInfo.Pt_j8     -> Pt_j8',
                            'EventInfo.Pt_j9     -> Pt_j9',
                            'EventInfo.Pt_j10     -> Pt_j10',
                            'EventInfo.Pt_j11     -> Pt_j11',
                            'EventInfo.Pt_j12     -> Pt_j12',
                            'EventInfo.Eta_j1    -> Eta_j1',
                            'EventInfo.Eta_j2    -> Eta_j2',
                            'EventInfo.Eta_j3    -> Eta_j3',
                            'EventInfo.Eta_j4    -> Eta_j4',
                            'EventInfo.Eta_j5    -> Eta_j5',
                            'EventInfo.Eta_j6    -> Eta_j6',
                            'EventInfo.Eta_j7    -> Eta_j7',
                            'EventInfo.Eta_j8    -> Eta_j8',
                            'EventInfo.Eta_j9    -> Eta_j9',
                            'EventInfo.Eta_j10    -> Eta_j10',
                            'EventInfo.Eta_j11    -> Eta_j11',
                            'EventInfo.Eta_j12    -> Eta_j12',
                            'EventInfo.Phi_j1    -> Phi_j1',
                            'EventInfo.Phi_j2    -> Phi_j2',
                            'EventInfo.Phi_j3    -> Phi_j3',
                            'EventInfo.Phi_j4    -> Phi_j4',
                            'EventInfo.Phi_j5    -> Phi_j5',
                            'EventInfo.Phi_j6    -> Phi_j6',
                            'EventInfo.Phi_j7    -> Phi_j7',
                            'EventInfo.Phi_j8    -> Phi_j8',
                            'EventInfo.Phi_j9    -> Phi_j9',
                            'EventInfo.Phi_j10    -> Phi_j10',
                            'EventInfo.Phi_j11    -> Phi_j11',
                            'EventInfo.Phi_j12    -> Phi_j12'
                              
                         ]
  algSeq += ntupleMaker

  treeFiller = createAlgorithm( 'CP::TreeFillerAlg', 'TreeFiller' )
  treeFiller.TreeName = treename
  algSeq += treeFiller

  return algSeq
