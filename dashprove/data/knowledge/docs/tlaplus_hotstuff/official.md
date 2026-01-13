# Computer Science > Distributed, Parallel, and Cluster Computing

**arXiv:1803.05069** (cs)
[Submitted on 13 Mar 2018 ([v1][1]), last revised 23 Jul 2019 (this version, v6)]

# Title:HotStuff: BFT Consensus in the Lens of Blockchain

Authors:[Maofan Yin][2], [Dahlia Malkhi][3], [Michael K. Reiter][4], [Guy Golan Gueta][5], [Ittai
Abraham][6]
View a PDF of the paper titled HotStuff: BFT Consensus in the Lens of Blockchain, by Maofan Yin and
4 other authors
[View PDF][7]

> Abstract:We present HotStuff, a leader-based Byzantine fault-tolerant replication protocol for the
> partially synchronous model. Once network communication becomes synchronous, HotStuff enables a
> correct leader to drive the protocol to consensus at the pace of actual (vs. maximum) network
> delay--a property called responsiveness--and with communication complexity that is linear in the
> number of replicas. To our knowledge, HotStuff is the first partially synchronous BFT replication
> protocol exhibiting these combined properties. HotStuff is built around a novel framework that
> forms a bridge between classical BFT foundations and blockchains. It allows the expression of
> other known protocols (DLS, PBFT, Tendermint, Casper), and ours, in a common framework.
> Our deployment of HotStuff over a network with over 100 replicas achieves throughput and latency
> comparable to that of BFT-SMaRt, while enjoying linear communication footprint during leader
> failover (vs. quadratic with BFT-SMaRt).

────┬───────────────────────────────────────────────────────────────────────────────────────────────
Comm│a shorter version of this paper has been published in PODC'19, which does not include          
ents│interpretation of other protocols using the framework, system evaluation or additional proofs  
:   │in appendices                                                                                  
────┼───────────────────────────────────────────────────────────────────────────────────────────────
Subj│Distributed, Parallel, and Cluster Computing (cs.DC)                                           
ects│                                                                                               
:   │                                                                                               
────┼───────────────────────────────────────────────────────────────────────────────────────────────
Cite│[arXiv:1803.05069][8] [cs.DC]                                                                  
as: │                                                                                               
────┼───────────────────────────────────────────────────────────────────────────────────────────────
    │(or [arXiv:1803.05069v6][9] [cs.DC] for this version)                                          
────┼───────────────────────────────────────────────────────────────────────────────────────────────
    │[https://doi.org/10.48550/arXiv.1803.05069][10]                                                
    │Focus to learn more                                                                            
    │arXiv-issued DOI via DataCite                                                                  
────┴───────────────────────────────────────────────────────────────────────────────────────────────

## Submission history

From: Maofan Yin [[view email][11]]
**[[v1]][12]** Tue, 13 Mar 2018 23:01:05 UTC (21 KB)
**[[v2]][13]** Thu, 18 Oct 2018 15:39:12 UTC (1,509 KB)
**[[v3]][14]** Mon, 18 Mar 2019 18:21:08 UTC (1,564 KB)
**[[v4]][15]** Tue, 2 Apr 2019 00:48:38 UTC (1,564 KB)
**[[v5]][16]** Wed, 5 Jun 2019 04:26:20 UTC (1,673 KB)
**[v6]** Tue, 23 Jul 2019 05:19:36 UTC (1,704 KB)
Full-text links:

## Access Paper:

* View a PDF of the paper titled HotStuff: BFT Consensus in the Lens of Blockchain, by Maofan Yin
  and 4 other authors
* [View PDF][17]
* [TeX Source ][18]
[view license][19]
Current browse context:
cs.DC
[< prev][20] | [next >][21]
[new][22] | [recent][23] | [2018-03][24]
Change to browse by:
[cs][25]

### References & Citations

* [NASA ADS][26]
* [Google Scholar][27]
* [Semantic Scholar][28]

### [DBLP][29] - CS Bibliography

[listing][30] | [bibtex][31]
[Ittai Abraham][32]
[Guy Gueta][33]
[Dahlia Malkhi][34]
export BibTeX citation Loading...

## BibTeX formatted citation

×
loading...
Data provided by:

### Bookmark

[ [BibSonomy logo] ][35] [ [Reddit logo] ][36]
Bibliographic Tools

# Bibliographic and Citation Tools

Bibliographic Explorer Toggle
Bibliographic Explorer *([What is the Explorer?][37])*
Connected Papers Toggle
Connected Papers *([What is Connected Papers?][38])*
Litmaps Toggle
Litmaps *([What is Litmaps?][39])*
scite.ai Toggle
scite Smart Citations *([What are Smart Citations?][40])*
Code, Data, Media

# Code, Data and Media Associated with this Article

alphaXiv Toggle
alphaXiv *([What is alphaXiv?][41])*
Links to Code Toggle
CatalyzeX Code Finder for Papers *([What is CatalyzeX?][42])*
DagsHub Toggle
DagsHub *([What is DagsHub?][43])*
GotitPub Toggle
Gotit.pub *([What is GotitPub?][44])*
Huggingface Toggle
Hugging Face *([What is Huggingface?][45])*
Links to Code Toggle
Papers with Code *([What is Papers with Code?][46])*
ScienceCast Toggle
ScienceCast *([What is ScienceCast?][47])*
Demos

# Demos

Replicate Toggle
Replicate *([What is Replicate?][48])*
Spaces Toggle
Hugging Face Spaces *([What is Spaces?][49])*
Spaces Toggle
TXYZ.AI *([What is TXYZ.AI?][50])*
Related Papers

# Recommenders and Search Tools

Link to Influence Flower
Influence Flower *([What are Influence Flowers?][51])*
Core recommender toggle
CORE Recommender *([What is CORE?][52])*

* Author
* Venue
* Institution
* Topic
About arXivLabs

# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly
on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of
openness, community, excellence, and user data privacy. arXiv is committed to these values and only
works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? [**Learn more about
arXivLabs**][53].

[Which authors of this paper are endorsers?][54] | [Disable MathJax][55] ([What is MathJax?][56])

[1]: https://arxiv.org/abs/1803.05069v1
[2]: https://arxiv.org/search/cs?searchtype=author&query=Yin,+M
[3]: https://arxiv.org/search/cs?searchtype=author&query=Malkhi,+D
[4]: https://arxiv.org/search/cs?searchtype=author&query=Reiter,+M+K
[5]: https://arxiv.org/search/cs?searchtype=author&query=Gueta,+G+G
[6]: https://arxiv.org/search/cs?searchtype=author&query=Abraham,+I
[7]: /pdf/1803.05069
[8]: https://arxiv.org/abs/1803.05069
[9]: https://arxiv.org/abs/1803.05069v6
[10]: https://doi.org/10.48550/arXiv.1803.05069
[11]: /show-email/30888bae/1803.05069
[12]: /abs/1803.05069v1
[13]: /abs/1803.05069v2
[14]: /abs/1803.05069v3
[15]: /abs/1803.05069v4
[16]: /abs/1803.05069v5
[17]: /pdf/1803.05069
[18]: /src/1803.05069
[19]: http://arxiv.org/licenses/nonexclusive-distrib/1.0/
[20]: /prevnext?id=1803.05069&function=prev&context=cs.DC
[21]: /prevnext?id=1803.05069&function=next&context=cs.DC
[22]: /list/cs.DC/new
[23]: /list/cs.DC/recent
[24]: /list/cs.DC/2018-03
[25]: /abs/1803.05069?context=cs
[26]: https://ui.adsabs.harvard.edu/abs/arXiv:1803.05069
[27]: https://scholar.google.com/scholar_lookup?arxiv_id=1803.05069
[28]: https://api.semanticscholar.org/arXiv:1803.05069
[29]: https://dblp.uni-trier.de
[30]: https://dblp.uni-trier.de/db/journals/corr/corr1803.html#abs-1803-05069
[31]: https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1803-05069
[32]: https://dblp.uni-trier.de/search/author?author=Ittai%20Abraham
[33]: https://dblp.uni-trier.de/search/author?author=Guy%20Gueta
[34]: https://dblp.uni-trier.de/search/author?author=Dahlia%20Malkhi
[35]: http://www.bibsonomy.org/BibtexHandler?requTask=upload&url=https://arxiv.org/abs/1803.05069&de
scription=HotStuff: BFT Consensus in the Lens of Blockchain
[36]: https://reddit.com/submit?url=https://arxiv.org/abs/1803.05069&title=HotStuff: BFT Consensus i
n the Lens of Blockchain
[37]: https://info.arxiv.org/labs/showcase.html#arxiv-bibliographic-explorer
[38]: https://www.connectedpapers.com/about
[39]: https://www.litmaps.co/
[40]: https://www.scite.ai/
[41]: https://alphaxiv.org/
[42]: https://www.catalyzex.com
[43]: https://dagshub.com/
[44]: http://gotit.pub/faq
[45]: https://huggingface.co/huggingface
[46]: https://paperswithcode.com/
[47]: https://sciencecast.org/welcome
[48]: https://replicate.com/docs/arxiv/about
[49]: https://huggingface.co/docs/hub/spaces
[50]: https://txyz.ai
[51]: https://influencemap.cmlab.dev/
[52]: https://core.ac.uk/services/recommender
[53]: https://info.arxiv.org/labs/index.html
[54]: /auth/show-endorsers/1803.05069
[55]: javascript:setMathjaxCookie()
[56]: https://info.arxiv.org/help/mathjax.html
