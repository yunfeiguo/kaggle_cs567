#!/usr/bin/env perl

for $i(@ARGV) {
    open IN,'<',$i or die "open($i):$!\n";
    while(<IN>){
	if (/: \d+/) {
	push @all,"$i,$_";
    }
    }
}
print $_,"\n" for (sort {$x=$1 if $a=~/: ([\d\.]+)/;$y=$1 if $b=~/: ([\d\.]+)/;return($x<=>$y)} @all);
