function values = readIeeeFloatsFromFile(filename)
    % Open file and read all lines
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file %s', filename);
    end
    
    % Read each line as a string
    lines = textscan(fid, '%s');
    fclose(fid);
    lines = lines{1};
    
    % Preallocate output
    n = numel(lines);
    values = zeros(n,1,'single');
    
    % Convert each 32-bit binary string to a single precision float
    for i = 1:n
        bitString = lines{i};
        if length(bitString) ~= 32
            error('Line %d does not have 32 bits', i);
        end
        u = uint32(bin2dec(bitString));   % binary string → integer
        values(i) = typecast(u, 'single'); % reinterpret integer → float
    end
end
