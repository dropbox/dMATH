# CryptoVerif: Cryptographic protocol verifier in the computational model

[CryptoVerif logo]
.

### Project leader:

[Bruno Blanchet][1]

### Project participants:

[Aymeric Fromherz][2], [Charlie Jacomme][3], [Benjamin Lipp][4]

### Former participant:

David Cadé
CryptoVerif is an automatic protocol prover sound in the computational model. It can prove

* secrecy;
* correspondences, which include in particular authentication;
* indistinguishability.
It provides a generic mechanism for specifying the security assumptions on cryptographic primitives,
which can handle in particular symmetric encryption, message authentication codes, public-key
encryption, signatures, hash functions.

The generated proofs are proofs by sequences of games, as used by cryptographers. These proofs are
valid for a number of sessions polynomial in the security parameter, in the presence of an active
adversary. CryptoVerif can also evaluate the probability of success of an attack against the
protocol as a function of the probability of breaking each cryptographic primitive and of the number
of sessions (exact security).

It also provides translations of protocol models to [OCaml][5] for generating implementations and to
[F*][6] for generating implementations and reasoning about them, and a translation of assumptions on
primitives to [EasyCrypt][7] for proving them from lower-level or more standard assumptions.

This prover is available below. This software is under development; please use it at your own risk.
Comments and bug reports welcome.

[Detailed description of CryptoVerif][8] (semantics of the language, security properties, game
transformations).

[Tutorial][9]

[Another tutorial][10], by Marc Hafner

[Another tutorial][11], by Charlie Jacomme and Benjamin Lipp

### Downloads

* Source: [CryptoVerif version 2.12, sources under the CeCILL-B license][12] (gzipped tar file)
* Binary: [CryptoVerif version 2.12 for Windows, binaries under the CeCILL-B license][13] (zip file)
* [Reference manual][14]

### Demo

An [online demo for CryptoVerif][15] is available.

### Mailing List

A mailing list is available for general discussions on CryptoVerif. I post announcements for new
releases of CryptoVerif on this list.

* To subscribe to the list, send an email to sympa-AT-inria.fr (replace -AT- with @) with subject
  "subscribe cryptoverif <your name>" (without quotes) or use this [web form][16].
* To post on the list, send an email to cryptoverif-AT-inria.fr (replace -AT- with @. Note: to avoid
  spam, you must subscribe to the list in order to be allowed to post.)

### Applications of CryptoVerif

Examples of application:

* [security proof of the FDH signature scheme][17] (with David Pointcheval)
* [Kerberos][18] (with Aaron D. Jaggard, Andre Scedrov, and Joe-Kai Tsay)
* [OEKE, One Encrypted Key Exchange][19]
* [SSH][20] (with David Cadé; scripts in subdirectory implementation/ssh of the CryptoVerif
  distribution)
* [Signal][21] (a.k.a. TextSecure, with Nadim Kobeissi and Karthikeyan Bhargavan)
* [TLS 1.3 Draft 18][22] (with Karthikeyan Bhargavan and Nadim Kobeissi)
* [ARINC 823 avionic protocols][23]
* [WireGuard][24] VPN protocol (with Benjamin Lipp and Karthikeyan Bhargavan)
* [Dynamic key compromise][25]
* [Translation of CryptoVerif assumptions to EasyCrypt][26] (with Pierre Boutry, Christian Doczkal,
  Benjamin Grégoire, Pierre-Yves Strub)
The files provided in the web pages above may correspond to older versions of CryptoVerif.
Up-to-date scripts for all these examples are included in the CryptoVerif distribution.

[References of papers that use CryptoVerif][27]

### Publications on this topic

* [1] *
  Bruno Blanchet. [Dealing with Dynamic Key Compromise in CryptoVerif][28]. In *Proceedings of the
  37th IEEE Computer Security Foundations Symposium (CSF'24)*, pages 495-510, Enschede, The
  Netherlands, July 2024. IEEE.
  
* [2] *
  Bruno Blanchet, Pierre Boutry, Christian Doczkal, Benjamin Grégoire, and Pierre-Yves Strub.
  [CV2EC: Getting the Best of Both Worlds][29]. In *Proceedings of the 37th IEEE Computer Security
  Foundations Symposium (CSF'24)*, pages 279-294, Enschede, The Netherlands, July 2024. IEEE.
  
* [3] *
  Bruno Blanchet and Charlie Jacomme. [Post-quantum sound CryptoVerif and verification of hybrid TLS
  and SSH key-exchanges][30]. In *Proceedings of the 37th IEEE Computer Security Foundations
  Symposium (CSF'24)*, pages 543-556, Enschede, The Netherlands, July 2024. IEEE.
  
* [4] *
  Bruno Blanchet. Cryptoverif: a computationally-sound security protocol verifier (initial version
  with communication on channels). Research report RR-9525, Inria, October 2023. Available at
  [https://inria.hal.science/hal-04246199][31].
  
* [5] *
  Bruno Blanchet and Charlie Jacomme. Cryptoverif: a computationally-sound security protocol
  verifier. Research report RR-9526, Inria, October 2023. Available at
  [https://inria.hal.science/hal-04253820][32].
  
* [6] *
  Joël Alwen, Bruno Blanchet, Eduard Hauck, Eike Kiltz, Benjamin Lipp, and Doreen Riepel. [Analysing
  the HPKE Standard][33]. In Anne Canteaut and Francois-Xavier Standaert, editors, * Eurocrypt
  2021*, volume 12696 of *Lecture Notes in Computer Science*, pages 87-116, Zagreb, Croatia, October
  2021. Springer.
  
* [7] *
  Joël Alwen, Bruno Blanchet, Eduard Hauck, Eike Kiltz, Benjamin Lipp, and Doreen Riepel. Analysing
  the HPKE standard. Cryptology ePrint Archive, Report 2020/1499, November 2020. Available at
  [https://eprint.iacr.org/2020/1499][34].
  
* [8] *
  Benjamin Lipp, Bruno Blanchet, and Karthikeyan Bhargavan. [A Mechanised Cryptographic Proof of the
  WireGuard Virtual Private Network Protocol][35]. In *IEEE European Symposium on Security and
  Privacy (EuroS&P'19)*, pages 231-246, Stockholm, Sweden, June 2019. IEEE Computer Society.
  
* [9] *
  Benjamin Lipp, Bruno Blanchet, and Karthikeyan Bhargavan. A mechanised cryptographic proof of the
  WireGuard virtual private network protocol. Research report RR-9269, Inria, April 2019. Available
  at [https://hal.inria.fr/hal-02100345][36].
  
* [10] *
  Bruno Blanchet. [Composition Theorems for CryptoVerif and Application to TLS 1.3][37]. In *31st
  IEEE Computer Security Foundations Symposium (CSF'18)*, pages 16-30, Oxford, UK, July 2018. IEEE
  Computer Society.
  
* [11] *
  Bruno Blanchet. Composition theorems for CryptoVerif and application to TLS 1.3. Research Report
  RR-9171, Inria, April 2018. Available at [https://hal.inria.fr/hal-01764527][38].
  
* [12] *
  Bruno Blanchet. [Symbolic and Computational Mechanized Verification of the ARINC823 Avionic
  Protocols][39]. In *30th IEEE Computer Security Foundations Symposium (CSF'17)*, pages 68-82,
  Santa Barbara, CA, USA, August 2017. IEEE.
  
* [13] *
  Karthikeyan Bhargavan, Bruno Blanchet, and Nadim Kobeissi. [Verified Models and Reference
  Implementations for the TLS 1.3 Standard Candidate][40]. In *IEEE Symposium on Security and
  Privacy (S&P'17)*, pages 483-503, San Jose, CA, May 2017. IEEE. Distinguished paper award.
  
* [14] *
  Karthikeyan Bhargavan, Bruno Blanchet, and Nadim Kobeissi. Verified models and reference
  implementations for the TLS 1.3 standard candidate. Research Report RR-9040, Inria, May 2017.
  Available at [https://hal.inria.fr/hal-01528752][41].
  
* [15] *
  Bruno Blanchet. Symbolic and computational mechanized verification of the ARINC823 avionic
  protocols. Research Report RR-9072, Inria, May 2017. Available at
  [https://hal.inria.fr/hal-01527671][42].
  
* [16] *
  Nadim Kobeissi, Karthikeyan Bhargavan, and Bruno Blanchet. [Automated Verification for Secure
  Messaging Protocols and their Implementations: A Symbolic and Computational Approach][43]. In *2nd
  IEEE European Symposium on Security and Privacy (EuroS&P'17)*, pages 435-450, Paris, France, April
  2017. IEEE.
  
* [17] *
  David Cadé and Bruno Blanchet. [Proved Generation of Implementations from Computationally Secure
  Protocol Specifications][44]. *Journal of Computer Security*, 23(3):331-402, 2015.
  
* [18] *
  David Cadé and Bruno Blanchet. [Proved Generation of Implementations from Computationally-Secure
  Protocol Specifications][45]. In David Basin and John Mitchell, editors, *2nd Conference on
  Principles of Security and Trust (POST 2013)*, volume 7796 of *Lecture Notes in Computer Science*,
  pages 63-82, Rome, Italy, March 2013. Springer.
  
* [19] *
  David Cadé and Bruno Blanchet. [From Computationally-Proved Protocol Specifications to
  Implementations and Application to SSH][46]. *Journal of Wireless Mobile Networks, Ubiquitous
  Computing, and Dependable Applications (JoWUA)*, 4(1):4-31, March 2013. Special issue ARES'12.
  
* [20] *
  David Cadé and Bruno Blanchet. [From Computationally-proved Protocol Specifications to
  Implementations][47]. In *7th International Conference on Availability, Reliability and Security
  (AReS 2012)*, pages 65-74, Prague, Czech Republic, August 2012. IEEE.
  
* [21] *
  Bruno Blanchet. [Automatically Verified Mechanized Proof of One-Encryption Key Exchange][48]. In
  *25th IEEE Computer Security Foundations Symposium (CSF'12)*, pages 325-339, Cambridge, MA, USA,
  June 2012. IEEE.
  
* [22] *
  Bruno Blanchet. [Mechanizing Game-Based Proofs of Security Protocols][49]. In Tobias Nipkow, Olga
  Grumberg, and Benedikt Hauptmann, editors, *Software Safety and Security - Tools for Analysis and
  Verification*, volume 33 of *NATO Science for Peace and Security Series - D: Information and
  Communication Security*, pages 1-25. IOS Press, May 2012. Proceedings of the summer school MOD
  2011.
  
* [23] *
  Bruno Blanchet. Automatically verified mechanized proof of one-encryption key exchange. Cryptology
  ePrint Archive, Report 2012/173, April 2012. Available at [http://eprint.iacr.org/2012/173][50].
  
* [24] *
  Bruno Blanchet. [A second look at Shoup's lemma][51]. In *Workshop on Formal and Computational
  Cryptography (FCC 2011)*, Paris, France, June 2011.
  
* [25] *
  Bruno Blanchet and David Pointcheval. [The computational and decisional Diffie-Hellman assumptions
  in CryptoVerif][52]. In *Workshop on Formal and Computational Cryptography (FCC 2010)*, Edimburgh,
  United Kingdom, July 2010.
  
* [26] *
  Bruno Blanchet, Aaron D. Jaggard, Jesse Rao, Andre Scedrov, and Joe-Kai Tsay. [Refining
  Computationally Sound Mechanized Proofs for Kerberos][53]. In *Workshop on Formal and
  Computational Cryptography (FCC 2009)*, Port Jefferson, NY, July 2009.
  
* [27] *
  Martín Abadi (invited speaker), Bruno Blanchet, and Hubert Comon-Lundh. [Models and Proofs of
  Protocol Security: A Progress Report][54]. In Ahmed Bouajjani and Oded Maler, editors, *21st
  International Conference on Computer Aided Verification (CAV'09)*, volume 5643 of * Lecture Notes
  in Computer Science*, pages 35-49, Grenoble, France, June 2009. Springer.
  
* [28] *
  Bruno Blanchet. *[Vérification automatique de protocoles cryptographiques : modèle formel et
  modèle calculatoire. Automatic verification of security protocols: formal model and computational
  model][55]*. Mémoire d'habilitation à diriger des recherches, Université Paris-Dauphine, November
  2008. En français avec publications en anglais en annexe. In French with publications in English
  in appendix.
  
* [29] *
  Bruno Blanchet, Aaron D. Jaggard, Andre Scedrov, and Joe-Kai Tsay. [Computationally Sound
  Mechanized Proofs for Basic and Public-key Kerberos][56]. In *ACM Symposium on Information,
  Computer and Communications Security (ASIACCS'08)*, pages 87-99, Tokyo, Japan, March 2008. ACM.
  
* [30] *
  Bruno Blanchet. [A Computationally Sound Mechanized Prover for Security Protocols][57]. *IEEE
  Transactions on Dependable and Secure Computing*, 5(4):193-207, October-December 2008. Special
  issue IEEE Symposium on Security and Privacy 2006. Electronic version available at
  [http://doi.ieeecomputersociety.org/10.1109/TDSC.2007.1005][58].
  
* [31] *
  Bruno Blanchet, Aaron D. Jaggard, Andre Scedrov, and Joe-Kai Tsay. [Computationally Sound
  Mechanized Proofs for Basic and Public-key Kerberos][59]. In *Dagstuhl seminar Formal Protocol
  Verification Applied*, October 2007.
  
* [32] *
  Bruno Blanchet. [CryptoVerif: A Computationally Sound Mechanized Prover for Cryptographic
  Protocols][60]. In *Dagstuhl seminar Formal Protocol Verification Applied*, October 2007.
  
* [33] *
  Bruno Blanchet. [Computationally Sound Mechanized Proofs of Correspondence Assertions][61]. In
  *20th IEEE Computer Security Foundations Symposium (CSF'07)*, pages 97-111, Venice, Italy, July
  2007. IEEE.
  
* [34] *
  Bruno Blanchet. Computationally sound mechanized proofs of correspondence assertions. Cryptology
  ePrint Archive, Report 2007/128, April 2007. Available at [http://eprint.iacr.org/2007/128][62].
  
* [35] *
  Bruno Blanchet and David Pointcheval. [Automated Security Proofs with Sequences of Games][63]. In
  Cynthia Dwork, editor, *CRYPTO'06*, volume 4117 of * Lecture Notes in Computer Science*, pages
  537-554, Santa Barbara, CA, August 2006. Springer.
  
* [36] *
  Bruno Blanchet. [A Computationally Sound Mechanized Prover for Security Protocols][64]. In *IEEE
  Symposium on Security and Privacy*, pages 140-154, Oakland, California, May 2006.
  
* [37] *
  Bruno Blanchet and David Pointcheval. Automated security proofs with sequences of games.
  Cryptology ePrint Archive, Report 2006/069, February 2006. Available at
  [http://eprint.iacr.org/2006/069][65].
  
* [38] *
  Bruno Blanchet. A computationally sound mechanized prover for security protocols. Cryptology
  ePrint Archive, Report 2005/401, November 2005. Available at
  [http://eprint.iacr.org/2005/401][66].
  
* [39] *
  Bruno Blanchet. [A Computationally Sound Automatic Prover for Cryptographic Protocols][67]. In
  *Workshop on the link between formal and computational models*, Paris, France, June 2005.
  

### Support of the CryptoVerif project

The development of CryptoVerif was partly supported by the ANR project [ARA SSIA Formacrypt][68] and
the ANR VERSO 2010 [ProSe][69]. [Bruno Blanchet][70]

[1]: https://bblanche.gitlabpages.inria.fr/
[2]: https://aymericfromherz.github.io/
[3]: https://charlie.jacomme.fr/
[4]: https://www.benjaminlipp.de/
[5]: https://ocaml.org
[6]: https://fstar-lang.org
[7]: https://easycrypt.info
[8]: https://inria.hal.science/hal-04253820v1/file/RR-9526.pdf
[9]: tutorial/index.html
[10]: https://rub-nds.github.io/AKE-Cryptoverif-Tutorial/Tutorial_mdbook/book/
[11]: https://github.com/charlie-j/summer-school-2023/tree/master
[12]: cryptoverif.html
[13]: cryptoverifbin.html
[14]: manual.pdf
[15]: http://proverif24.paris.inria.fr/cryptoverif.php
[16]: https://sympa.inria.fr/sympa/subscribe/cryptoverif
[17]: FDH
[18]: kerberos
[19]: OEKE/
[20]: https://bblanche.gitlabpages.inria.fr/publications/CadeBlanchetJoWUA13.html
[21]: https://github.com/Inria-Prosecco/proscript-messaging
[22]: https://github.com/inria-prosecco/reftls
[23]: https://bblanche.gitlabpages.inria.fr/arinc823/
[24]: WireGuard
[25]: compromise/
[26]: cv2EasyCrypt/
[27]: cryptoverif-users.html
[28]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetCSF24.html
[29]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetBoutryetalCSF24.html
[30]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetJacommeCSF24.html
[31]: https://inria.hal.science/hal-04246199
[32]: https://inria.hal.science/hal-04253820
[33]: http://bblanche.gitlabpages.inria.fr/publications/AlwenetalEurocrypt21.html
[34]: https://eprint.iacr.org/2020/1499
[35]: http://bblanche.gitlabpages.inria.fr/publications/LippBlanchetBhargavanEuroSP19.html
[36]: https://hal.inria.fr/hal-02100345
[37]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetCSF18.html
[38]: https://hal.inria.fr/hal-01764527
[39]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetCSF17.html
[40]: http://bblanche.gitlabpages.inria.fr/publications/BhargavanBlanchetKobeissiSP2017.html
[41]: https://hal.inria.fr/hal-01528752
[42]: https://hal.inria.fr/hal-01527671
[43]: http://bblanche.gitlabpages.inria.fr/publications/KobeissiBhargavanBlanchetEuroSP17.html
[44]: http://bblanche.gitlabpages.inria.fr/publications/CadeBlanchetJCS14.html
[45]: http://bblanche.gitlabpages.inria.fr/publications/CadeBlanchetPOST13.html
[46]: http://bblanche.gitlabpages.inria.fr/publications/CadeBlanchetJoWUA13.html
[47]: http://bblanche.gitlabpages.inria.fr/publications/CadeBlanchetARES12.html
[48]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetCSF12.html
[49]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetMOD11.html
[50]: http://eprint.iacr.org/2012/173
[51]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetFCC11.html
[52]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetPointchevalFCC10.html
[53]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetJaggardRaoScedrovTsayFCC09.html
[54]: http://bblanche.gitlabpages.inria.fr/publications/AbadiBlanchetComonCAV09.html
[55]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetHDR.html
[56]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetJaggardScedrovTsayAsiaCCS08.html
[57]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetTDSC07.html
[58]: http://doi.ieeecomputersociety.org/10.1109/TDSC.2007.1005
[59]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetJaggardScedrovTsayDagstuhl07.html
[60]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetDagstuhl07.html
[61]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetCSF07.html
[62]: http://eprint.iacr.org/2007/128
[63]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetPointchevalCrypto06.html
[64]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetOakland06.html
[65]: http://eprint.iacr.org/2006/069
[66]: http://eprint.iacr.org/2005/401
[67]: http://bblanche.gitlabpages.inria.fr/publications/BlanchetLink05.html
[68]: https://bblanche.gitlabpages.inria.fr/formacrypt/index.html
[69]: https://crypto.di.ens.fr/projects:prose:main
[70]: https://bblanche.gitlabpages.inria.fr/
