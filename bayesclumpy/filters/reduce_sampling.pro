; Reduce the sampling of the filters to allow a faster
; execution of BayesCLUMPY. If a filter contains less
; number of wavelength points, leave it untouched
; reduce_sampling, 200
pro reduce_sampling, nlambda_final
	files = file_search('ORIGINAL/*.res')
	nf = n_elements(files)

; Loop over all files
	for i = 0, nf-1 do begin
		
; Extract name of the filter
		res = strsplit(files[i],'/',/extract)
		new_name = res[1]
		
		openr,2,files[i]
		readf,2,nlambda,normalization

; If the number of wavelength points is larger than the target, reintepolate and save the new file
		if (nlambda gt nlambda_final) then begin
			dat = dblarr(2,nlambda)
			readf,2,dat

			x = dindgen(nlambda_final) / (nlambda_final-1.d0) * (max(dat[0,*])-min(dat[0,*])) + min(dat[0,*])
			t = interpol(dat[1,*], dat[0,*], x)

; Clean negative values
			ind = where(t lt 0.d0, count)
			if (count ne 0) then begin
				t[ind] = abs(t[ind])
			endif
			
			openw,3,new_name
			printf,3,nlambda_final, normalization
			for j = 0, nlambda_final-1 do printf,3,x[j],t[j]
			close,3

			print, new_name, nlambda, ' -> ', nlambda_final
		endif else begin

; Else just copy the file
			file_copy, files[i], new_name, /overwrite
			print, new_name, ' - just copied'
		endelse
		close,2
	endfor

	stop
end