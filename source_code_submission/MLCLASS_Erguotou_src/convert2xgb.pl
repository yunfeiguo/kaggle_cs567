#!/usr/bin/env perl
#input
#Id,minutes_past,radardist_km,Ref,RefComposite,RhoHV,Zdr,Kdp,Expected,MarshallPalmer,Katsumata,Brandes,Sachidanazrnic,RyzhkovZrnic
#2,29.0833333333,2.0,16.625,22.6666666667,0.998611114167,0.380208333333,-0.265297071393,1.0160005,0.630814032327,0.671864467267,0.534997905388,-15.0993330428,-40.0380701552
#output
#1 101:1.2 102:0.03
#label idx:value idx:value

while(<>){
    chomp;
    if ($.==1) {
	@header = split /,/;
    } else{
	@f=split /,/;
	@out = ();
	for $i(0..$#f) {
	    next if $header[$i] eq 'Id';
	    if($header[$i] eq 'Expected') {
		#$label = ($f[$i] > 100)? 1:0;
		#unshift @out,$label;
		unshift @out,$f[$i];
	    } else {
		push @out,$i.":".$f[$i];
	    }
	}
	print join(" ",@out),"\n";
    }
}
warn "conversion to XGBoost format done\n";
