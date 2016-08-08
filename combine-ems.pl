#!/usr/bin/perl
# OneDrive/Research/ISM-models/SILCC/bin/optimal_post_processing

$prefix = $ARGV[0];
@FILES = glob($prefix."*.ems");

$first = $FILES[0];

open(IN, "<", $first);
chomp(@data = <IN>);
close(IN);

$header = $data[0];

for $file (@FILES){
    print STDERR "Processing $file\n";
    $ID = substr($file,length($prefix),(length($file)-(length(".ems")+length($prefix)) ));
    open(IN, "<", $file);
    chomp(@data = <IN>);
    if($data[0] eq $header){
	
	# Here we push the entire emissivity data, which is found on rows 1 and greater. In the case
	# one zone models this is simply the second entry in data.

	# Calculate average of emissivity file
	$AverageData = &avgEmissivty(\@data);

	# Join ID number and emissivity into a string.
	my $string = join("\t",($ID,@AverageData));

	# Push model string into full suite of data
	push(@full_data,$string);
    }
    elsif($data[0] ne $header){exit "Header miss-match\n";}
    close(IN);
}

open(OUT, ">", $prefix."combined-ems.tbl");

@header_array = split("\t",$header);
$header_array[0]="depth";
unshift(@header_array,"ID");
@spacer_array = @header_array;

for ($i=0; $i<=  scalar @header_array -1; $i++){    
    $spacer_array[$i] =  "-"x(length($header_array[$i]));
}
print OUT join("\t",@header_array)."\n";
print OUT join("\t",@spacer_array)."\n";

for $outline (@full_data){
    print OUT $outline."\n";
}
close(OUT);


sub avgEmissivty{
    # dereference array containing emissivity data
    my $array_ref = $_[0];
    my @array_rows = @$array_ref;

    # Initialize depth at the face of the cloud
    $r_old = 0.0;

    # Loop over each row
    
    #Read number of header entries, split by tab
    my @heade_array = split("\t",$array_rows[0]);

    # Initialize SumProduct and AverageEmissivity arrays
    @SumProduct_dr_emissivity = (0) x scalar @header_array;
    @AverageEmissivity = (0) x scalar @header_array;

    $number_of_rows = scalar @array_rows -1;
    print STDERR "rows 1 to ${number_of_rows}\n";

    for (my $row = 1; $row <= @array_rows -1; $row++){
        
        @array_columns = split("\t",$array_rows[$row]);


        $r_old = $r_new;
        $r_new = $array_columns[0];
        $dr = $r_new - $r_old;

        $Sum_dr = $r_new; # the variable r is the depth into the cloud. It is the total r, so it represents the sum of all drs.

        for(my $col = 1; $col <= @array_columns -1;$col++){
            $SumProduct_dr_emissivity[$col] += $dr * $array_columns[$col];
        }
    }

    for (my $col = 1; $col <= @SumProduct_dr_emissivity; $i++){
        $AverageEmissivity[$col] = $SumProduct_dr_emissivity[$col] / $Sum_dr;
    }
    return(join("\t",@AverageEmissivity))
}
