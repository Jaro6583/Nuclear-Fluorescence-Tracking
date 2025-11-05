function giveError( lhs, qual, rhs, andOr, errormsg)

if length( lhs) ~= length( rhs) || length( lhs) ~= length( rhs) || length(lhs) ~= length( qual)
    error( 'something went wrong' )
end
conds = {};
len = length( lhs);

for jj = 1 : len

    conds = { conds{:}; lhs{jj}, qual{jj}, rhs{jj} };

end

for jj = 1 : len
    eval( ['res( ', num2str(jj), ') = ' num2str(conds{jj,1}), conds{jj,2}, num2str( conds{jj,3}) , ';'] );
end
if strcmp( andOr, 'and')
    if all(res)
        error(errormsg)       
    end
elseif strcmp( andOr, 'or')
    if any(res)
        error( errormsg)
    end
end



end
