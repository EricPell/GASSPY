#!/usr/bin/perl

$prefix = $ARGV[0];
@FILES = glob($prefix."*.ems");

$first = $FILES[0];

open(IN, "<", $first);
chomp(@data = <IN>);
close(IN);

$header = $data[0];

for $file (@FILES){
    $ID = substr($file,length($prefix),(length($file)-(length(".ems")+length($prefix)) ));
    open(IN, "<", $file);
    chomp(@data = <IN>);
    if($data[0] eq $header){
	
	# Here we push the entire emissivity data, which is found on rows 1 and greater. In the case
	# one zone models this is simply the second entry in data.

	# Calculate average of emissivity file
	@AverageData = &avgEmissivty(\@data);

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
close(OUT)


sub avgEmissivty{
    my @array = @{$_[0]};
    
}
