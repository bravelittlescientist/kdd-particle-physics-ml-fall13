# CS273a Fall 2013 - Final Project

A machine learning and data mining project based on the KDD Particle Physics dataset.

[Project Description](http://sli.ics.uci.edu/Classes/2013F-273a?action=download&upname=Project.pdf)

[KDD 2004 Challenge](http://osmot.cs.cornell.edu/kddcup/)

## Design

[scikit learn, ML for python](http://scikit-learn.org/stable/)

## Data

[KDD Source](http://osmot.cs.cornell.edu/kddcup/datasets.html)

[Physics Data - Original Archive](https://www.dropbox.com/s/bnjutmo2mv7g6xg/data_kddcup04.tar.gz)

Github does not play well with large datasets, so we will maintain versions of our data in tarballs.

Repository Structure
    
    /
        data/
            phy_raw.tgz

To unpack:

    $ cd data && tar -zxf phy_raw.tar.gz

Then,
    
    /
        data/
            phy_raw.tgz
            raw/
                phy_test.dat
                phy_train.dat
