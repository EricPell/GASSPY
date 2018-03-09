#!/usr/bin/perl -w
use Getopt::Long;
use File::Find::Rule;

GetOptions ("j=i" => \$j,#number of processors 
	    "mask=s" => \$mask_file);

if(defined($mask_file)){
    open (MASK,"<$mask_file");
    chomp(@mask_files = <MASK>);    
    foreach $mask (@mask_files){
	$mask{$mask}=1;
    }
}

$os = `uname`;
chomp($os);

chomp($shellname = `echo \$SHELL`);
if ($shellname eq "/bin/bash"){
    $do_source = 0;
    chomp($CLOUDY_PATH = `echo \$CLOUDY_PATH`);
    chomp($CLOUDY_EXE  = `echo \$CLOUDY_EXE`);
    if(length($CLOUDY_PATH) == 0){
	print "CLOUDY_PATH not defined. Please specify:";
	chomp($CLOUDY_PATH = <STDIN>);
	print "Append `export CLOUDY_PATH=\"${CLOUDY_PATH}\"` to your .bashrc file? (y/n)";
	chomp($answer = <STDIN>);
	if ($answer eq "y"){
	    system "echo \"export CLOUDY_PATH=\'${CLOUDY_PATH}\'\" >> ~/.bashrc";
	    $do_source = 1;
	}
	else{print "You did not answer 'y'. Fine, do it yourself. But until you do, you will keep getting this message.\n"}
    }
    if(length($CLOUDY_EXE) == 0){
	print "CLOUDY_EXE not defined. Please specify(d for default cloudy.exe):";
	chomp($CLOUDY_EXE = <STDIN>);
	if($CLOUDY_EXE eq "d"){$CLOUDY_EXE = "cloudy.exe"}
	print "Append `export CLOUDY_EXE=\"${CLOUDY_EXE}\"` to your .bashrc file? (y/n)";
	chomp($answer = <STDIN>);
	if ($answer eq "y"){
	    system "echo \"export CLOUDY_EXE=\'${CLOUDY_EXE}\'\" >> ~/.bashrc";
	    $do_source = 1;
	}
	else{print "You did not answer 'y'. Fine, do it yourself. But until you do, you will keep getting this message.\n"}
    }

    if($do_source == 1){
	system "source ~/.bashrc";
    }
}
else{
    die("Your shell is not bash. At the moment, I don't support running out of bash")
}


if (defined($ARGV[0])){
    $stuff=$ARGV[0];
}
else{
	#Searching ./cloudy-output/ for model files
	$stuff="./cloudy-output/*.in";
}
    
# this loops over all the *.in files in the current directory
# and runs the code to produce *.out files

{
	#@input_files = File::Find::Rule->in("$stuff");
    @input_files = glob("$stuff");
    $i = 0;
    
    if (defined($j)) {
	while ($i <= @input_files -1)
	{
	    @_ = `ps -e | grep ${CLOUDY_EXE}`;
	    $num_proc= scalar @_;
	    
	    if($num_proc < $j){
		$input = $input_files[$i];
		$output = $input;
		$output =~ s/\.in//gi;
		$out = "$output".".out";
		
		if(defined($mask{$out})){
		    print "skipping run model $input\n";
		}
		else{
		    $remaining = @input_files -1 - $i;
		    print( "model $i, remaining models:$remaining\n$input --> $out\n\n" );
		    
		    # actually execute the code
		    system "nice -n 5 ${CLOUDY_PATH}/${CLOUDY_EXE} < $input  > $out&";
		    {sleep(0.25);}			
		}
		
		$i++;
		
	    }
	    else {sleep(1);}
	}
    }
    
    else{
	foreach $input ( @input_files ){
	    $output = $input;
	    $output =~ s/\.in//gi;
	    $out = "$output".".out";
	    print( "$input going to $out\n" );
	    # actually execute the code
	    system "nice -n 5 ${CLOUDY_PATH}/${CLOUDY_EXE} < $input  > $out";	
	}
    }

	@_ = `ps -e | grep ${CLOUDY_EXE}`;
	$num_proc= scalar @_;

	while (1 <= $num_proc)
	{
		print("sleeping, still have $num_proc to run....Zzzz....Zzzzz\r");
		sleep(1);
	    @_ = `ps -e | grep ${CLOUDY_EXE}`;
	    $num_proc= scalar @_;
	}
}

