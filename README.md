# Datavoids in Knowledge Graphs



# Installation and Running Instructions

This repo requires access toi a GPU-enabled compute instance. 




## Code Organization Explanation:

```helpers/```

```Kelpie_package/```


## Supported flows:  

[] Flow 1: KG data preparation and high-level analysis  
- give initial stats on the kg in question
    - # of triples, # of enities, highest degree node, lowest degree node. 
    - how connected is the kg??
- show relations with its relation types 
    - so that the user can choose relations to then do the next step. 
[] Data void curation
- using the relations chosen above, run ```get_candidates.sh```
- mdify so that we sabe all of the intermediayte smaller kgs generated and tested. 
- maybe would have to change how to choose the head and tail for a given relation?? 


[] Data for simulation preparation
[] Simulation run
[] Visualization

