function A = readHexFloatFile(filename)
%READHEXFLOATFILE Reads a text file of IEEE-754 single-precision floats in hex
%
%   A = readHexFloatFile(filename)
%
%   Each line in the file should contain hexadecimal values (8 hex digits per float)
%   separated by spaces. The output A is a matrix of single-precision floats.
%
%   Example file contents:
%       3f800000 40000000 40400000
%       40800000 40a00000 40c00000
%
%   Example usage:
%       A = readHexFloatFile('data.txt');

    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    data = {};
    lineIndex = 0;

    while true
        tline = fgetl(fid);
        if ~ischar(tline)
            break; % end of file
        end

        tline = strtrim(tline);
        if isempty(tline)
            continue; % skip empty lines
        end

        % Split the line into hexadecimal tokens
        hexVals = strsplit(tline);

        % Convert to uint32 then reinterpret as single
        floats = typecast(uint32(hex2dec(hexVals)), 'single');

        lineIndex = lineIndex + 1;
        data{lineIndex, 1} = floats;
    end

    fclose(fid);

    % Combine all rows into a single matrix
    A = cell2mat(data);
end
