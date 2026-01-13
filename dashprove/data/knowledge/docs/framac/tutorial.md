[Frama-C][1]

* [Features][2]
* [Documentation][3]
* [Publications][4]
* [Blog][5]
* [Jobs][6]
* [Contact][7]
* [Download][8]
[******][9]

# Tutorials

# Frama-C Framework

Julien Signoles
Enseigner "Frama-C pour la cybersécurité" : retour d'expérience [[link][10]]
In Rendez-vous de la Recherche et de l'Enseignement de la Sécurité des Systèmes d'Information
(RESSI), 2025
*In French*
*

Cet article présente un retour d’expérience lié à un enseignement de Frama-C pour la cybersécurité
dispensé dans plusieurs formations d’ingénieurs et universitaires de niveau master. Il est constitué
d’un cours et d’un TP de trois heures chacun. Frama-C est une plateforme open source fournissant des
analyseurs de code C. Le cours est articulé autour de ses trois techniques et analyseurs principaux,
à savoir la vérification déductive avec Wp, l’interprétation abstraite avec Eva, et la vérification
d’annotations à l’exécution avec E-ACSL. Le but du cours est de faire découvrir ces techniques
formelles aux étudiants en insistant sur leurs aspects pratiques en cybersécurité.

*
Allan Blanchard, Nikolai Kosmatov, Frédéric Loulergue
A Lesson on Verification of IoT Software with Frama-C [[link][11]]
In International Conference on High Performance Computing & Simulation (HPCS), 2019
*

This paper is a tutorial introduction to Frama-C, a framework for the analysis and verification of
sequential C programs, and in particular its EVA, WP, and E-ACSL plugins. The examples are drawn
from Contiki, a lightweight operating system for the Internet of Things.

*
Nikolai Kosmatov, Julien Signoles
Frama-C, a Collaborative Framework for C Code Verification. Tutorial Synopsis [[link][12]]
In International Conference on Runtime Verification (RV), 2016
*

Frama-C is a source code analysis platform that aims at conducting verification of industrial-size C
programs. It provides its users with a collection of plug-ins that perform static and dynamic
analysis for safety- and security-critical software. Collaborative verification across cooperating
plug-ins is enabled by their integration on top of a shared kernel, and their compliance to a common
specification language, ACSL.

This paper presents a three-hour tutorial on Frama-C in which we provide a comprehensive overview of
its most important plug-ins: the abstract-interpretation based plug-in Value, the deductive
verification tool WP, the runtime verification tool E-ACSL and the test generation tool PathCrawler.
We also emphasize different possible collaborations between these plug-ins and a few others. The
presentation is illustrated on concrete examples of C programs

*
Virgile Prevosto
Frama-C tutorial
In STANCE Project, 2013
*

* Frama-C Overview: [Slides][13], [Simple plugin][14]
* Value Analysis: [Slides][15] and [Examples][16]
* WP: [Slides][17] and [Examples][18]
*
David Mentré
Practical introduction to Frama-C [[link][19]]
In Mitsubishi Electric R&D Centre Europe, 2013
*

[Examples][20]

*

[1]: /index.html
[2]: /html/kernel-plugin.html
[3]: /html/documentation.html
[4]: /html/publications.html
[5]: /blog/index.html
[6]: /html/jobs.html
[7]: /html/contact.html
[8]: /html/get-frama-c.html
[9]: /html/get-frama-c.html
[10]: https://julien-signoles.fr/publis/2025_ressi.pdf
[11]: https://hal.inria.fr/hal-02317078/en
[12]: https://julien-signoles.fr/publis/2016_rv.pdf
[13]: /download/publications/2013-stance-p/frama_c_overview.pdf
[14]: /download/publications/2013-stance-p/simple_metrics.ml
[15]: /download/publications/2013-stance-p/value_tutorial.pdf
[16]: /download/publications/2013-stance-p/value_examples.tar.gz
[17]: /download/publications/2013-stance-p/wp_advanced_slides.pdf
[18]: /download/publications/2013-stance-p/wp_examples.tar.gz
[19]: /download/publications/2013-merce-m/introduction-to-frama-c_v2.pdf
[20]: /download/publications/2013-merce-m/introduction-slides-examples.tar.gz
