#!/usr/bin/perl

#@groups=(0, 64, 128, 192, 256, 384, 512);
@groups=(64, 128, 192, 256, 384, 512);
@groupSize=(64, 128, 192, 256, 384, 512, 1024);


foreach $g (@groups) {
    foreach $gs (@groupSize) {
        $f = "hipstream.float.$g.$gs";
        $cmd = "./gpu-stream-hip --float --groups $g --groupSize $gs";
        print "Run $f : $cmd\n";
         
        system "$cmd > $f";

    }
}

foreach $g (@groups) {
    foreach $gs (@groupSize) {
        $f = "hipstream.double.$g.$gs";
        $cmd = "./gpu-stream-hip --groups $g --groupSize $gs";
        print "Run $f : $cmd\n";
         
        system "$cmd > $f";

    }
}
