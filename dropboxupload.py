import dropbox

dbx = dropbox.Dropbox('enteryouraccountdetailshere')
dbx.users_get_current_account()
dbx.files_upload("violation", '/C:/TraViol/licenseplate.png')
