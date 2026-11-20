# Failure examples (test split)

## Lexicon rule (edit distance)  —  1013 false positives, 471 false negatives

**False Negatives**

- p(offensive)=0.000 · true=Offensive · “I am not here to be constructive, I am here to prove that Wikipedia is nothing but a bunch of hypocrites and liars.”
- p(offensive)=0.000 · true=Offensive · “Drop dead   Drop dead.”
- p(offensive)=0.000 · true=Offensive · “Actually, you are the cockroach that followed me to the notice board, and repeatedly comes back to revert what I had written. FYI.207.161.232.45”
- p(offensive)=0.000 · true=Offensive · “NOW! REMOVE IT FROM YOUR SITE NOW! YOU DAMNED THIEVES!”
- p(offensive)=0.000 · true=Offensive · “I think you're being a hypocrite. 207.97.213.169”

**False Positives**

- p(offensive)=1.000 · true=Clean · “very often, I find that others are on the ball already. Here's something you'll probably find useful -)”
- p(offensive)=1.000 · true=Clean · “Aha, closer perusal suggests that I can use Template:PD-old-70.”
- p(offensive)=1.000 · true=Clean · “My favourite kiddie is Josh H as he and I are very fond of cucumber destruction. (anus)”
- p(offensive)=1.000 · true=Clean · “What Do You Mean?   I'm curious what do you mean? I'm open for cooperation in the matter. What do you propose and thx for not being a jerk about it.  mcelite”
- p(offensive)=1.000 · true=Clean · “for example i'm from Central italy(Marche) and i have light brown hair and blue-green eyes , i'm white or black?”

## TF-IDF + LogReg  —  290 false positives, 330 false negatives

**False Negatives**

- p(offensive)=0.006 · true=Offensive · “fake barnstars and add them to other editors pages.  This is deceptive and disruptive.  Thanks!”
- p(offensive)=0.011 · true=Offensive · “Already left it on the other page   SO EITHER DISCUSS IT OR BE BANNED FROM WIKIPEDIA.”
- p(offensive)=0.012 · true=Offensive · “Current Squad   No FOOKIN AROUND WITH IT, alright!”
- p(offensive)=0.020 · true=Offensive · “Removing warnings is considered anger that other people type bull**** on my discussion page. Just for your info.”
- p(offensive)=0.022 · true=Offensive · “I count four (4) rabid members of the furluminati, intent on crushing all opposing points of view, assuming that 1 or more are not simply sockpuppets.”

**False Positives**

- p(offensive)=1.000 · true=Clean · “LOL PENIS!   LOL!  I AM TALKING TO YOU SO THIS SHIT IS NOT VANDALISING!”
- p(offensive)=0.999 · true=Clean · “Copyright MXMXLI BY ASS_JH GY}}”
- p(offensive)=0.999 · true=Clean · “Khurki is a dumb Assk Jeeves?”
- p(offensive)=0.999 · true=Clean · “Your a disgrace Dan stop making up silly stuff about the wombats”
- p(offensive)=0.994 · true=Clean · “guess what   your retarded”

## Word2Vec + FFNN (SGD)  —  328 false positives, 443 false negatives

**False Negatives**

- p(offensive)=0.011 · true=Offensive · “Already left it on the other page   SO EITHER DISCUSS IT OR BE BANNED FROM WIKIPEDIA.”
- p(offensive)=0.015 · true=Offensive · “I count four (4) rabid members of the furluminati, intent on crushing all opposing points of view, assuming that 1 or more are not simply sockpuppets.”
- p(offensive)=0.022 · true=Offensive · “contador   G O  F U C K  Y O U R S E L F  Love dc”
- p(offensive)=0.039 · true=Offensive · “Materials   What are they using to contain whatever they're smuggling? Condoms and seran wrap?”
- p(offensive)=0.044 · true=Offensive · “I've requested a change to Wolfpuss - congratulations to all dirty thinkers in our community. (!) I submit to your naughtiness.”

**False Positives**

- p(offensive)=1.000 · true=Clean · “Copyright MXMXLI BY ASS_JH GY}}”
- p(offensive)=1.000 · true=Clean · “LOL PENIS!   LOL!  I AM TALKING TO YOU SO THIS SHIT IS NOT VANDALISING!”
- p(offensive)=0.999 · true=Clean · “Khurki is a dumb Assk Jeeves?”
- p(offensive)=0.999 · true=Clean · “Not at this time, fag.”
- p(offensive)=0.998 · true=Clean · “Is Crones that disease where you constantly have to go poo? Ever soiled your pants?”

## BERT-base (fp32)  —  459 false positives, 373 false negatives

**False Negatives**

- p(offensive)=0.020 · true=Offensive · “I AM ADDING REFERENCES AND AM REMOVING THE TAGS AS I GO. SO STICK YOUR LAST WARNING!!!”
- p(offensive)=0.022 · true=Offensive · “Stop writing on my T.P. I have nothing further to discuss with you given that you lack a bsinc education or diplomacy”
- p(offensive)=0.026 · true=Offensive · “Look here, you can't sock puppet this this image in to avoid the 3RR. You've been warned. Read WP:3RR”
- p(offensive)=0.026 · true=Offensive · “This is for removing my post on 100%   I'm going to DDOS your toaster for this.”
- p(offensive)=0.028 · true=Offensive · “Already left it on the other page   SO EITHER DISCUSS IT OR BE BANNED FROM WIKIPEDIA.”

**False Positives**

- p(offensive)=0.975 · true=Clean · “Wil Wheaton Wil Wheaton loves star wars you jerk”
- p(offensive)=0.974 · true=Clean · “Listen pal, I do things my way, and you think your so hotsie totsie, too bad yoou're not getting inside me.”
- p(offensive)=0.974 · true=Clean · “friends   I and Jxy are two damned friends who use the same damned computer! Take it easy! Ciao,”
- p(offensive)=0.973 · true=Clean · “=WOT A SELF-Important, pompous ballbag”
- p(offensive)=0.971 · true=Clean · “your hot dog vendor will get you one day. gooooo pancreatic bad-ness!”

## BERT-base (dynamic INT8)  —  333 false positives, 492 false negatives

**False Negatives**

- p(offensive)=0.056 · true=Offensive · “Stop writing on my T.P. I have nothing further to discuss with you given that you lack a bsinc education or diplomacy”
- p(offensive)=0.062 · true=Offensive · “fake barnstars and add them to other editors pages.  This is deceptive and disruptive.  Thanks!”
- p(offensive)=0.066 · true=Offensive · “Wikistalker and sock puppet Lay off.  I will have recourse.  You are not immune.   Stan”
- p(offensive)=0.073 · true=Offensive · “Censorship and abuse of oversight.   Please die very slowly and painfully.”
- p(offensive)=0.076 · true=Offensive · “Tiptoety your judgemts seems to consits of he;s been blocked before, so must be guilty. That indicates a very dull mind.”

**False Positives**

- p(offensive)=0.909 · true=Clean · “Listen pal, I do things my way, and you think your so hotsie totsie, too bad yoou're not getting inside me.”
- p(offensive)=0.904 · true=Clean · “Wil Wheaton Wil Wheaton loves star wars you jerk”
- p(offensive)=0.904 · true=Clean · “Don't threaten me pov warrior.”
- p(offensive)=0.890 · true=Clean · “What the hell happened to this dude?”
- p(offensive)=0.879 · true=Clean · “Argh  Berserk button. Get a better catchphrase ,fruitcake.”
