import astropy.table as at


data = at.Table.read('/mnt/ceph/users/gaia/dr3/csv/RvsMeanSpectrum/RvsMeanSpectrum_000000-003111.csv.gz', format='ecsv')