<u>Understand the problem</u>

1. Overview

   1) What is a protein

   Proteins are the workhorses of the cell, carrying out a vast array of tasks, from catalyzing chemical reactions to transporting molecules across membranes to transmitting signals between cells. Understanding how proteins work and what they do is essential for understanding the basic processes of life, as well as for developing new drugs, designing enzymes for industrial applications, and much more. Proteins are large molecules built of 20 different types of building blocks called amino acids. And while knowing the amino acid sequence (or just "sequence") of a protein is relatively simple, inferring what is does, or its function is still an open problem in computational molecular biology.

   Therefore, one of the most pressing challenges is to accurately predict  the function of newly discovered proteins based on their amino-acid sequence. This is where computational methods come in. By using algorithms to analyze protein sequences and structures, we can make educated guesses  about what a protein might do, based on its similarity to other proteins with known functions.

   Based on the protein sequence of amino acids, participants are asked to predict Gene Ontology (GO) terms in each of the three subontologies:
   - Molecular Function (MF)
   - Biological Process  (BP)
   - Cellular Component (CC)

   X = 'MNSVTVSHAPYTITYHDDWEPVMSQLVEFYNEVASWLLRDETSPIPDKFFIQLKQPLRNKRVCVCGIDPYPKDGTGVPFESPNFTKKSIKEIASSISRLTGVIDYKGYNLNIIDGVIPWNYYLSCKLGETKSHAIYWDKISKLLLQHITKHVSVLYCLGKTDFSNIRAKLESPVTTIVGYHPAARDRQFEKDRSFEIINVLLELDNKVPINWAQGFIY'

   y = GO(MF,BP,CC)

   **<u>Important note</u> **:  *GO terms are hierarchical.*
