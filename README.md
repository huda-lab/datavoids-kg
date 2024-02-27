# Datavoids in Knowledge Graphs



# Installation and Running Instructions

This repo requires access to a GPU-enabled compute instance. 

To setup 




## Code Organization Explanation:
TODO: explain the contents of each python file. 

```helpers/```

```Kelpie_package/```


## Supported flows
The following flows are intended to be run in sequential fashion, each building on top of each other.   

## Flow 1: KG high-level analysis  
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

