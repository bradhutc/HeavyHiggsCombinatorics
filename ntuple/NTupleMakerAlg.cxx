#include <AsgTools/MessageCheckAsgTools.h>
#include <MuonTagger/NTupleMakerAlg.h>

#include <xAODMissingET/MissingETContainer.h>
#include "xAODMissingET/MissingETComposition.h"
#include "xAODMissingET/MissingETAuxContainer.h"
#include "xAODMissingET/MissingETAssociationMap.h"

#include <xAODTruth/TruthParticleContainer.h>
#include <xAODEventInfo/EventInfo.h>

#include <xAODJet/JetContainer.h>
#include <TH1.h>

namespace {
  const static SG::AuxElement::ConstAccessor<unsigned int> origin("classifierParticleOrigin");

  const static SG::AuxElement::Decorator<float> Pt_bj1("Pt_bj1");
  const static SG::AuxElement::Decorator<float> Pt_bj2("Pt_bj2");
  const static SG::AuxElement::Decorator<float> Pt_bj3("Pt_bj3");
  const static SG::AuxElement::Decorator<float> Pt_bj4("Pt_bj4");
  const static SG::AuxElement::Decorator<float> Pt_bj5("Pt_bj5");
  const static SG::AuxElement::Decorator<float> Pt_bj6("Pt_bj6");
  const static SG::AuxElement::Decorator<float> Eta_bj1("Eta_bj1");
  const static SG::AuxElement::Decorator<float> Eta_bj2("Eta_bj2");
  const static SG::AuxElement::Decorator<float> Eta_bj3("Eta_bj3");
  const static SG::AuxElement::Decorator<float> Eta_bj4("Eta_bj4");
  const static SG::AuxElement::Decorator<float> Eta_bj5("Eta_bj5");
  const static SG::AuxElement::Decorator<float> Eta_bj6("Eta_bj6");
  const static SG::AuxElement::Decorator<float> Phi_bj1("Phi_bj1");
  const static SG::AuxElement::Decorator<float> Phi_bj2("Phi_bj2");
  const static SG::AuxElement::Decorator<float> Phi_bj3("Phi_bj3");
  const static SG::AuxElement::Decorator<float> Phi_bj4("Phi_bj4");
  const static SG::AuxElement::Decorator<float> Phi_bj5("Phi_bj5");
  const static SG::AuxElement::Decorator<float> Phi_bj6("Phi_bj6");

  const static SG::AuxElement::Decorator<float> M_b4_1("M_b4_1");
  const static SG::AuxElement::Decorator<float> Pt_b4_1("Pt_b4_1");
  const static SG::AuxElement::Decorator<float> Eta_b4_1("Eta_b4_1");
  const static SG::AuxElement::Decorator<float> Phi_b4_1("Phi_b4_1");
  const static SG::AuxElement::Decorator<float> M_b4_2("M_b4_2");
  const static SG::AuxElement::Decorator<float> Pt_b4_2("Pt_b4_2");
  const static SG::AuxElement::Decorator<float> Eta_b4_2("Eta_b4_2");
  const static SG::AuxElement::Decorator<float> Phi_b4_2("Phi_b4_2");

  const static SG::AuxElement::Decorator<float> M_H1("M_H1");
  const static SG::AuxElement::Decorator<float> Pt_H1("Pt_H1");
  const static SG::AuxElement::Decorator<float> Eta_H1("Eta_H1");
  const static SG::AuxElement::Decorator<float> Phi_H1("Phi_H1");
  const static SG::AuxElement::Decorator<float> M_H2("M_H2");
  const static SG::AuxElement::Decorator<float> Pt_H2("Pt_H2");
  const static SG::AuxElement::Decorator<float> Eta_H2("Eta_H2");
  const static SG::AuxElement::Decorator<float> Phi_H2("Phi_H2");

  



  const static SG::AuxElement::Decorator<float> Pt_j1("Pt_j1");
  const static SG::AuxElement::Decorator<float> Pt_j2("Pt_j2");
  const static SG::AuxElement::Decorator<float> Pt_j3("Pt_j3");
  const static SG::AuxElement::Decorator<float> Pt_j4("Pt_j4");
  const static SG::AuxElement::Decorator<float> Pt_j5("Pt_j5");
  const static SG::AuxElement::Decorator<float> Pt_j6("Pt_j6");
  const static SG::AuxElement::Decorator<float> Pt_j7("Pt_j7");
  const static SG::AuxElement::Decorator<float> Pt_j8("Pt_j8");
  const static SG::AuxElement::Decorator<float> Pt_j9("Pt_j9");
  const static SG::AuxElement::Decorator<float> Pt_j10("Pt_j10");
  const static SG::AuxElement::Decorator<float> Pt_j11("Pt_j11");
  const static SG::AuxElement::Decorator<float> Pt_j12("Pt_j12");
  const static SG::AuxElement::Decorator<float> Eta_j1("Eta_j1");
  const static SG::AuxElement::Decorator<float> Eta_j2("Eta_j2");
  const static SG::AuxElement::Decorator<float> Eta_j3("Eta_j3");
  const static SG::AuxElement::Decorator<float> Eta_j4("Eta_j4");
  const static SG::AuxElement::Decorator<float> Eta_j5("Eta_j5");
  const static SG::AuxElement::Decorator<float> Eta_j6("Eta_j6");
  const static SG::AuxElement::Decorator<float> Eta_j7("Eta_j7");
  const static SG::AuxElement::Decorator<float> Eta_j8("Eta_j8");
  const static SG::AuxElement::Decorator<float> Eta_j9("Eta_j9");
  const static SG::AuxElement::Decorator<float> Eta_j10("Eta_j10");
  const static SG::AuxElement::Decorator<float> Eta_j11("Eta_j11");
  const static SG::AuxElement::Decorator<float> Eta_j12("Eta_j12");
  const static SG::AuxElement::Decorator<float> Phi_j1("Phi_j1");
  const static SG::AuxElement::Decorator<float> Phi_j2("Phi_j2");
  const static SG::AuxElement::Decorator<float> Phi_j3("Phi_j3");
  const static SG::AuxElement::Decorator<float> Phi_j4("Phi_j4");
  const static SG::AuxElement::Decorator<float> Phi_j5("Phi_j5");
  const static SG::AuxElement::Decorator<float> Phi_j6("Phi_j6");
  const static SG::AuxElement::Decorator<float> Phi_j7("Phi_j7");
  const static SG::AuxElement::Decorator<float> Phi_j8("Phi_j8");
  const static SG::AuxElement::Decorator<float> Phi_j9("Phi_j9");
  const static SG::AuxElement::Decorator<float> Phi_j10("Phi_j10");
  const static SG::AuxElement::Decorator<float> Phi_j11("Phi_j11");
  const static SG::AuxElement::Decorator<float> Phi_j12("Phi_j12");

  const static SG::AuxElement::Decorator<float> HTb("HTb");
  const static SG::AuxElement::Decorator<float> HT("HT");
  const static SG::AuxElement::Decorator<int> num_jets("num_jets");
  const static SG::AuxElement::Decorator<int> num_bjets("num_bjets");
  const static SG::AuxElement::Decorator<float> met_NonInt("met_NonInt");
  const static SG::AuxElement::Decorator<float> set_NonInt("set_NonInt");
   

}

NTupleMakerAlg::NTupleMakerAlg(const std::string& name, ISvcLocator* pSvcLocator)
  : EL::AnaAlgorithm(name, pSvcLocator)
{
}

StatusCode NTupleMakerAlg::initialize()
{

  ANA_CHECK(m_systematicsList.initialize());

  ANA_CHECK(book(TH1F("bj1_pt", "bj1_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("bj2_pt", "bj2_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("bj3_pt", "bj3_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("bj4_pt", "bj4_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("bj5_pt", "bj5_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("bj6_pt", "bj6_pt", 200, 0, 0)));

  ANA_CHECK(book(TH1F("bj1_eta", "bj1_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj2_eta", "bj2_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj3_eta", "bj3_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj4_eta", "bj4_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj5_eta", "bj5_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj6_eta", "bj6_eta", 200, -5, 5)));

  ANA_CHECK(book(TH1F("bj1_phi", "bj1_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj2_phi", "bj2_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj3_phi", "bj3_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj4_phi", "bj4_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj5_phi", "bj5_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("bj6_phi", "bj6_phi", 200, -5, 5)));

  ANA_CHECK(book(TH1F("j1_pt", "j1_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j2_pt", "j2_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j3_pt", "j3_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j4_pt", "j4_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j5_pt", "j5_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j6_pt", "j6_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j7_pt", "j7_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j8_pt", "j8_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j9_pt", "j9_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j10_pt", "j10_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j11_pt", "j11_pt", 200, 0, 0)));
  ANA_CHECK(book(TH1F("j12_pt", "j12_pt", 200, 0, 0)));

  ANA_CHECK(book(TH1F("j1_eta", "j1_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j2_eta", "j2_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j3_eta", "j3_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j4_eta", "j4_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j5_eta", "j5_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j6_eta", "j6_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j7_eta", "j7_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j8_eta", "j8_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j9_eta", "j9_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j10_eta", "j10_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j11_eta", "j11_eta", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j12_eta", "j12_eta", 200, -5, 5)));

  ANA_CHECK(book(TH1F("j1_phi", "j1_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j2_phi", "j2_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j3_phi", "j3_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j4_phi", "j4_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j5_phi", "j5_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j6_phi", "j6_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j7_phi", "j7_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j8_phi", "j8_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j9_phi", "j9_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j10_phi", "j10_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j11_phi", "j11_phi", 200, -5, 5)));
  ANA_CHECK(book(TH1F("j12_phi", "j12_phi", 200, -5, 5)));

  ANA_CHECK(book(TH1F("M_b4_1", "M_b4_1", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Pt_b4_1", "Pt_b4_1", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Eta_b4_1", "Eta_b4_1", 200, -5, 5)));
  ANA_CHECK(book(TH1F("Phi_b4_1", "Phi_b4_1", 200, -5, 5)));
  ANA_CHECK(book(TH1F("M_b4_2", "M_b4_2", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Pt_b4_2", "Pt_b4_2", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Eta_b4_2", "Eta_b4_2", 200, -5, 5)));
  ANA_CHECK(book(TH1F("Phi_b4_2", "Phi_b4_2", 200, -5, 5)));
  ANA_CHECK(book(TH1F("M_H1", "M_H1", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Pt_H1", "Pt_H1", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Eta_H1", "Eta_H1", 200, -5, 5)));
  ANA_CHECK(book(TH1F("Phi_H1", "Phi_H1", 200, -5, 5)));
  ANA_CHECK(book(TH1F("M_H2", "M_H2", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Pt_H2", "Pt_H2", 200, 0, 0)));
  ANA_CHECK(book(TH1F("Eta_H2", "Eta_H2", 200, -5, 5)));
  ANA_CHECK(book(TH1F("Phi_H2", "Phi_H2", 200, -5, 5)));


  ANA_CHECK(book(TH1F("HTb", "HTb", 200, 0, 0)));
  ANA_CHECK(book(TH1F("HT", "HT", 200, 0, 0)))

  ANA_CHECK(book(TH1F("num_bjets", "num_bjets", 200, 0, 0)));
  ANA_CHECK(book(TH1F("num_jets", "num_jets", 200, 0, 0)));
  ANA_CHECK(book(TH1F("met_nonint", "met_nonint", 200, 0, 200)));
  ANA_CHECK(book(TH1F("set_nonint", "set_nonint", 200, 0, 200)));
  // ANA_CHECK(book(TH1F("Pt_jets", "Pt_jets", 200, 0, 1000)));


  return StatusCode::SUCCESS;
}

StatusCode NTupleMakerAlg::execute()
{

  const xAOD::EventInfo* eventInfo = nullptr;
  ANA_CHECK(evtStore()->retrieve(eventInfo, "EventInfo"));

// ----------------------------------------------------

  const xAOD::TruthParticleContainer* bottoms = nullptr;
  ANA_CHECK(evtStore()->retrieve(bottoms, "TruthBottom"));

  const xAOD::JetContainer* truthjets = nullptr;
  ANA_CHECK( evtStore()->retrieve (truthjets, "AntiKt4TruthDressedWZJets"));

  const xAOD::JetContainer* truthlargeRjets = nullptr;
  ANA_CHECK( evtStore()->retrieve (truthlargeRjets, "AntiKt10TruthSoftDropBeta100Zcut10Jets"));
 
  const xAOD::TruthParticleContainer* BSMs = nullptr;
  ANA_CHECK(evtStore()->retrieve(BSMs, "TruthBSM"));
// ----------------------------------------------------

  const xAOD::MissingETContainer* mets = nullptr;
  ANA_CHECK (evtStore()->retrieve (mets, "MET_Truth"));
  ANA_MSG_INFO("--> NonInt MET = " << (*mets)["NonInt"]->met());
  met_NonInt(*eventInfo) = (*mets)["NonInt"]->met();


  const xAOD::MissingETContainer* sets = nullptr;
  ANA_CHECK (evtStore()->retrieve (sets, "MET_Truth"));
  ANA_MSG_INFO("--> NonInt SET = " << (*sets)["NonInt"]->sumet());
  set_NonInt(*eventInfo) = (*sets)["Int"]->sumet();

// ----------------------------------------------------

  const xAOD::JetContainer* jets = nullptr;
  ANA_CHECK(evtStore()->retrieve(jets, "AntiKt4TruthDressedWZJets"));
 

auto TrueBottoms = std::make_unique<ConstDataVector<xAOD::TruthParticleContainer>>(SG::VIEW_ELEMENTS);
auto VLQ = std::make_unique<ConstDataVector<xAOD::TruthParticleContainer>>(SG::VIEW_ELEMENTS);
auto HeavyHiggs = std::make_unique<ConstDataVector<xAOD::TruthParticleContainer>>(SG::VIEW_ELEMENTS);
for (const auto bottom : *bottoms) {
    if (bottom->nParents() > 0){
        const xAOD::TruthParticle* parentBottom = bottom->parent(0);
        if (!parentBottom) continue; // Skip this iteration if parentBottom is nullptr
        int parentID = parentBottom->absPdgId();
        ANA_MSG_INFO("Parent ID: " << parentID);
    //     if (parentID == 5) {
    //     TrueBottoms->push_back(bottom);
    // } else if (parentID == 7) {
    //     VLQ->push_back(parentBottom);
    // } else if (parentID == 35) { 
    //     HeavyHiggs->push_back(parentBottom);
    // }
    if (parentID == 7) {
        VLQ->push_back(parentBottom);
        TrueBottoms->push_back(bottom);
    } else if (parentID == 35) { 
        HeavyHiggs->push_back(parentBottom);
        TrueBottoms->push_back(bottom);
    }
  }
}

if (TrueBottoms->size() < 3 || TrueBottoms->size() > 6) { //changed from 6 to 2-6
  ANA_CHECK(setDefaults());
  ANA_MSG_DEBUG("Didn't find between 3 and 6 bjets");
  return StatusCode::SUCCESS;
}


double cHTb = 0.0; //Initialize total scalar Et for bjets
auto bjets = std::make_unique<ConstDataVector<xAOD::JetContainer>>(SG::VIEW_ELEMENTS);

for (auto jet : *jets) {
  // Check if the jet is a b-jet
  bool isBJet = false;
  for (const auto& bottom : *TrueBottoms) {
    double dR = jet->p4().DeltaR(bottom->p4());
      if (dR < 0.4) {
          isBJet = true;
          cHTb += (jet->pt() * cosh(jet->eta())); //0.001 factor to convert MeV to GeV
          break;
      }
    }
    if (isBJet) {
        bjets->push_back(jet);
    }
  }

double cHT = 0.0; //Initialize total scalar Et
for (auto jet : *jets) {
  cHT += (jet->pt() * cosh(jet->eta())); //0.001 factor to convert MeV to GeV
}


HTb(*eventInfo) = cHTb;
HT(*eventInfo) = cHT;

TLorentzVector b4_1 = VLQ->size() > 0 ? VLQ->at(0)->p4() : TLorentzVector();
TLorentzVector b4_2 = VLQ->size() > 1 ? VLQ->at(1)->p4() : TLorentzVector();
TLorentzVector h1 = HeavyHiggs->size() > 2 ? HeavyHiggs->at(2)->p4() : TLorentzVector();
TLorentzVector h2 = HeavyHiggs->size() > 3 ? HeavyHiggs->at(3)->p4() : TLorentzVector();

M_b4_1(*eventInfo) = b4_1.M() * 0.001;
Pt_b4_1(*eventInfo) = b4_1.Pt() * 0.001;
Eta_b4_1(*eventInfo) = b4_1.Eta();
Phi_b4_1(*eventInfo) = b4_1.Phi();

M_b4_2(*eventInfo) = b4_2.M() * 0.001;
Pt_b4_2(*eventInfo) = b4_2.Pt() * 0.001;
Eta_b4_2(*eventInfo) = b4_2.Eta();
Phi_b4_2(*eventInfo) = b4_2.Phi();


M_H1(*eventInfo) = h1.M() * 0.001;
Pt_H1(*eventInfo) = h1.Pt() * 0.001;
Eta_H1(*eventInfo) = h1.Eta();
Phi_H1(*eventInfo) = h1.Phi();

M_H2(*eventInfo) = h2.M() * 0.001;
Pt_H2(*eventInfo) = h2.Pt() * 0.001;
Eta_H2(*eventInfo) = h2.Eta();
Phi_H2(*eventInfo) = h2.Phi();


TLorentzVector bj1 = bjets->size() > 0 ? bjets->at(0)->p4() : TLorentzVector();
TLorentzVector bj2 = bjets->size() > 1 ? bjets->at(1)->p4() : TLorentzVector();
TLorentzVector bj3 = bjets->size() > 2 ? bjets->at(2)->p4() : TLorentzVector();
TLorentzVector bj4 = bjets->size() > 3 ? bjets->at(3)->p4() : TLorentzVector();
TLorentzVector bj5 = bjets->size() > 4 ? bjets->at(4)->p4() : TLorentzVector();
TLorentzVector bj6 = bjets->size() > 5 ? bjets->at(5)->p4() : TLorentzVector();

TLorentzVector j1 = jets->size() > 0 ? jets->at(0)->p4() : TLorentzVector();
TLorentzVector j2 = jets->size() > 1 ? jets->at(1)->p4() : TLorentzVector();
TLorentzVector j3 = jets->size() > 2 ? jets->at(2)->p4() : TLorentzVector();
TLorentzVector j4 = jets->size() > 3 ? jets->at(3)->p4() : TLorentzVector();
TLorentzVector j5 = jets->size() > 4 ? jets->at(4)->p4() : TLorentzVector();
TLorentzVector j6 = jets->size() > 5 ? jets->at(5)->p4() : TLorentzVector();
TLorentzVector j7 = jets->size() > 6 ? jets->at(6)->p4() : TLorentzVector();
TLorentzVector j8 = jets->size() > 7 ? jets->at(7)->p4() : TLorentzVector();
TLorentzVector j9 = jets->size() > 8 ? jets->at(8)->p4() : TLorentzVector();
TLorentzVector j10 = jets->size() > 9 ? jets->at(9)->p4() : TLorentzVector();
TLorentzVector j11 = jets->size() > 10 ? jets->at(10)->p4() : TLorentzVector();
TLorentzVector j12 = jets->size() > 11 ? jets->at(11)->p4() : TLorentzVector();


num_bjets(*eventInfo) = bjets->size();
num_jets(*eventInfo) = jets->size();


Pt_bj1(*eventInfo) = bj1.Pt() * 0.001; // Convert MeV to GeV
Pt_bj2(*eventInfo) = bj2.Pt() * 0.001;
Pt_bj3(*eventInfo) = bj3.Pt() * 0.001;
Pt_bj4(*eventInfo) = bj4.Pt() * 0.001;
Pt_bj5(*eventInfo) = bj5.Pt() * 0.001;
Pt_bj6(*eventInfo) = bj6.Pt() * 0.001;

Eta_bj1(*eventInfo) = bj1.Eta();
Eta_bj2(*eventInfo) = bj2.Eta();
Eta_bj3(*eventInfo) = bj3.Eta();
Eta_bj4(*eventInfo) = bj4.Eta();
Eta_bj5(*eventInfo) = bj5.Eta();
Eta_bj6(*eventInfo) = bj6.Eta();

Phi_bj1(*eventInfo) = bj1.Phi();
Phi_bj2(*eventInfo) = bj2.Phi();
Phi_bj3(*eventInfo) = bj3.Phi();
Phi_bj4(*eventInfo) = bj4.Phi();
Phi_bj5(*eventInfo) = bj5.Phi();
Phi_bj6(*eventInfo) = bj6.Phi();



Pt_j1(*eventInfo) = j1.Pt() * 0.001;
Pt_j2(*eventInfo) = j2.Pt() * 0.001;
Pt_j3(*eventInfo) = j3.Pt() * 0.001;
Pt_j4(*eventInfo) = j4.Pt() * 0.001;
Pt_j5(*eventInfo) = j5.Pt() * 0.001;
Pt_j6(*eventInfo) = j6.Pt() * 0.001;
Pt_j7(*eventInfo) = j7.Pt() * 0.001;
Pt_j8(*eventInfo) = j8.Pt() * 0.001;
Pt_j9(*eventInfo) = j9.Pt() * 0.001;
Pt_j10(*eventInfo) = j10.Pt() * 0.001;
Pt_j11(*eventInfo) = j11.Pt() * 0.001;
Pt_j12(*eventInfo) = j12.Pt() * 0.001;

Eta_j1(*eventInfo) = j1.Eta();
Eta_j2(*eventInfo) = j2.Eta();
Eta_j3(*eventInfo) = j3.Eta();
Eta_j4(*eventInfo) = j4.Eta();
Eta_j5(*eventInfo) = j5.Eta();
Eta_j6(*eventInfo) = j6.Eta();
Eta_j7(*eventInfo) = j7.Eta();
Eta_j8(*eventInfo) = j8.Eta();
Eta_j9(*eventInfo) = j9.Eta();
Eta_j10(*eventInfo) = j10.Eta();
Eta_j11(*eventInfo) = j11.Eta();
Eta_j12(*eventInfo) = j12.Eta();

Phi_j1(*eventInfo) = j1.Phi();
Phi_j2(*eventInfo) = j2.Phi();
Phi_j3(*eventInfo) = j3.Phi();
Phi_j4(*eventInfo) = j4.Phi();
Phi_j5(*eventInfo) = j5.Phi();
Phi_j6(*eventInfo) = j6.Phi();
Phi_j7(*eventInfo) = j7.Phi();
Phi_j8(*eventInfo) = j8.Phi();
Phi_j9(*eventInfo) = j9.Phi();
Phi_j10(*eventInfo) = j10.Phi();
Phi_j11(*eventInfo) = j11.Phi();
Phi_j12(*eventInfo) = j12.Phi();



  // Convert units from MeV to GeV
  met_NonInt(*eventInfo) *= 0.001;
  set_NonInt(*eventInfo) *= 0.001;
  HTb(*eventInfo) *= 0.001; // Convert MeV to GeV
  HT(*eventInfo) *= 0.001; // Convert MeV to GeV


  hist("M_b4_1")->Fill(M_b4_1(*eventInfo));
  hist("Pt_b4_1")->Fill(Pt_b4_1(*eventInfo));
  hist("Eta_b4_1")->Fill(Eta_b4_1(*eventInfo));
  hist("Phi_b4_1")->Fill(Phi_b4_1(*eventInfo));
  hist("M_b4_2")->Fill(M_b4_2(*eventInfo));
  hist("Pt_b4_2")->Fill(Pt_b4_2(*eventInfo));
  hist("Eta_b4_2")->Fill(Eta_b4_2(*eventInfo));
  hist("Phi_b4_2")->Fill(Phi_b4_2(*eventInfo));

  hist("M_H1")->Fill(M_H1(*eventInfo));
  hist("Pt_H1")->Fill(Pt_H1(*eventInfo));
  hist("Eta_H1")->Fill(Eta_H1(*eventInfo));
  hist("Phi_H1")->Fill(Phi_H1(*eventInfo));
  hist("M_H2")->Fill(M_H2(*eventInfo));
  hist("Pt_H2")->Fill(Pt_H2(*eventInfo));
  hist("Eta_H2")->Fill(Eta_H2(*eventInfo));
  hist("Phi_H2")->Fill(Phi_H2(*eventInfo));


  hist("bj1_pt")->Fill(bj1.Pt());
  hist("bj2_pt")->Fill(bj2.Pt());
  hist("bj3_pt")->Fill(bj3.Pt());
  hist("bj4_pt")->Fill(bj4.Pt());
  hist("bj5_pt")->Fill(bj5.Pt());
  hist("bj6_pt")->Fill(bj6.Pt());

  
  hist("bj1_eta")->Fill(bj1.Eta());
  hist("bj2_eta")->Fill(bj2.Eta());
  hist("bj3_eta")->Fill(bj3.Eta());
  hist("bj4_eta")->Fill(bj4.Eta());
  hist("bj5_eta")->Fill(bj5.Eta());
  hist("bj6_eta")->Fill(bj6.Eta());


  hist("bj1_phi")->Fill(bj1.Phi());
  hist("bj2_phi")->Fill(bj2.Phi());
  hist("bj3_phi")->Fill(bj3.Phi());
  hist("bj4_phi")->Fill(bj4.Phi());
  hist("bj5_phi")->Fill(bj5.Phi());
  hist("bj6_phi")->Fill(bj6.Phi());



  hist("j1_pt")->Fill(j1.Pt());
  hist("j2_pt")->Fill(j2.Pt());
  hist("j3_pt")->Fill(j3.Pt());
  hist("j4_pt")->Fill(j4.Pt());
  hist("j5_pt")->Fill(j5.Pt());
  hist("j6_pt")->Fill(j6.Pt());
  hist("j7_pt")->Fill(j7.Pt());
  hist("j8_pt")->Fill(j8.Pt());
  hist("j9_pt")->Fill(j9.Pt());
  hist("j10_pt")->Fill(j10.Pt());
  hist("j11_pt")->Fill(j11.Pt());
  hist("j12_pt")->Fill(j12.Pt());

  hist("j1_eta")->Fill(j1.Eta());
  hist("j2_eta")->Fill(j2.Eta());
  hist("j3_eta")->Fill(j3.Eta());
  hist("j4_eta")->Fill(j4.Eta());
  hist("j5_eta")->Fill(j5.Eta());
  hist("j6_eta")->Fill(j6.Eta());
  hist("j7_eta")->Fill(j7.Eta());
  hist("j8_eta")->Fill(j8.Eta());
  hist("j9_eta")->Fill(j9.Eta());
  hist("j10_eta")->Fill(j10.Eta());
  hist("j11_eta")->Fill(j11.Eta());
  hist("j12_eta")->Fill(j12.Eta());

  hist("j1_phi")->Fill(j1.Phi());
  hist("j2_phi")->Fill(j2.Phi());
  hist("j3_phi")->Fill(j3.Phi());
  hist("j4_phi")->Fill(j4.Phi());
  hist("j5_phi")->Fill(j5.Phi());
  hist("j6_phi")->Fill(j6.Phi());
  hist("j7_phi")->Fill(j7.Phi());
  hist("j8_phi")->Fill(j8.Phi());
  hist("j9_phi")->Fill(j9.Phi());
  hist("j10_phi")->Fill(j10.Phi());
  hist("j11_phi")->Fill(j11.Phi());
  hist("j12_phi")->Fill(j12.Phi());


  hist("met_nonint")->Fill(met_NonInt(*eventInfo));
  hist("set_nonint")->Fill(set_NonInt(*eventInfo));

  hist("num_bjets")->Fill(num_bjets(*eventInfo));
  hist("num_jets")->Fill(num_jets(*eventInfo));


  hist("HTb")->Fill(HTb(*eventInfo));
  hist("HT")->Fill(HT(*eventInfo));

  return StatusCode::SUCCESS;
}

StatusCode NTupleMakerAlg::setDefaults()
{
  const xAOD::EventInfo* eventInfo = nullptr;
  ANA_CHECK(evtStore()->retrieve(eventInfo, "EventInfo"));

  M_b4_1(*eventInfo) = -999;
  Pt_b4_1(*eventInfo) = -999;
  Eta_b4_1(*eventInfo) = -999;
  Phi_b4_1(*eventInfo) = -999;
  M_b4_2(*eventInfo) = -999;
  Pt_b4_2(*eventInfo) = -999;
  Eta_b4_2(*eventInfo) = -999;
  Phi_b4_2(*eventInfo) = -999;

  M_H1(*eventInfo) = -999;
  Pt_H1(*eventInfo) = -999;
  Eta_H1(*eventInfo) = -999;
  Phi_H1(*eventInfo) = -999;
  M_H2(*eventInfo) = -999;
  Pt_H2(*eventInfo) = -999;
  Eta_H2(*eventInfo) = -999;
  Phi_H2(*eventInfo) = -999;

  // Basic angular variables
  Pt_bj1(*eventInfo) = -999;
  Pt_bj2(*eventInfo) = -999;
  Pt_bj3(*eventInfo) = -999;
  Pt_bj4(*eventInfo) = -999;
  Pt_bj5(*eventInfo) = -999;
  Pt_bj6(*eventInfo) = -999;

  Eta_bj1(*eventInfo) = -999;
  Eta_bj2(*eventInfo) = -999;
  Eta_bj3(*eventInfo) = -999;
  Eta_bj4(*eventInfo) = -999;
  Eta_bj5(*eventInfo) = -999;
  Eta_bj6(*eventInfo) = -999;

  Phi_bj1(*eventInfo) = -999;
  Phi_bj2(*eventInfo) = -999;
  Phi_bj3(*eventInfo) = -999;
  Phi_bj4(*eventInfo) = -999;
  Phi_bj5(*eventInfo) = -999;
  Phi_bj6(*eventInfo) = -999;

  Pt_j1(*eventInfo) = -999;
  Pt_j2(*eventInfo) = -999;
  Pt_j3(*eventInfo) = -999;
  Pt_j4(*eventInfo) = -999;
  Pt_j5(*eventInfo) = -999;
  Pt_j6(*eventInfo) = -999;
  Pt_j7(*eventInfo) = -999;
  Pt_j8(*eventInfo) = -999;
  Pt_j9(*eventInfo) = -999;
  Pt_j10(*eventInfo) = -999;
  Pt_j11(*eventInfo) = -999;
  Pt_j12(*eventInfo) = -999;

  Eta_j1(*eventInfo) = -999;
  Eta_j2(*eventInfo) = -999;
  Eta_j3(*eventInfo) = -999;
  Eta_j4(*eventInfo) = -999;
  Eta_j5(*eventInfo) = -999;
  Eta_j6(*eventInfo) = -999;
  Eta_j7(*eventInfo) = -999;
  Eta_j8(*eventInfo) = -999;
  Eta_j9(*eventInfo) = -999;
  Eta_j10(*eventInfo) = -999;
  Eta_j11(*eventInfo) = -999;
  Eta_j12(*eventInfo) = -999;

  Phi_j1(*eventInfo) = -999;
  Phi_j2(*eventInfo) = -999;
  Phi_j3(*eventInfo) = -999;
  Phi_j4(*eventInfo) = -999;
  Phi_j5(*eventInfo) = -999;
  Phi_j6(*eventInfo) = -999;
  Phi_j7(*eventInfo) = -999;
  Phi_j8(*eventInfo) = -999;
  Phi_j9(*eventInfo) = -999;
  Phi_j10(*eventInfo) = -999;
  Phi_j11(*eventInfo) = -999;
  Phi_j12(*eventInfo) = -999;
  

  num_jets(*eventInfo) = -999;


  HTb(*eventInfo) = -999;
  HT(*eventInfo) = -999;

  met_NonInt(*eventInfo) = -999;
  set_NonInt(*eventInfo) = -999;
  num_bjets(*eventInfo) = -999;

  return StatusCode::SUCCESS;
}

StatusCode NTupleMakerAlg::finalize()
{

  return StatusCode::SUCCESS;
}
