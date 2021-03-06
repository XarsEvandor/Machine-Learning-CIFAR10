??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
?
ccnn_custom/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*)
shared_nameccnn_custom/dense/kernel
?
,ccnn_custom/dense/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/dense/kernel*
_output_shapes
:	?
*
dtype0
?
ccnn_custom/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameccnn_custom/dense/bias
}
*ccnn_custom/dense/bias/Read/ReadVariableOpReadVariableOpccnn_custom/dense/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
ccnn_custom/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameccnn_custom/conv2d/kernel
?
-ccnn_custom/conv2d/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/conv2d/kernel*&
_output_shapes
:	*
dtype0
?
%ccnn_custom/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%ccnn_custom/batch_normalization/gamma
?
9ccnn_custom/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp%ccnn_custom/batch_normalization/gamma*
_output_shapes
:	*
dtype0
?
$ccnn_custom/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$ccnn_custom/batch_normalization/beta
?
8ccnn_custom/batch_normalization/beta/Read/ReadVariableOpReadVariableOp$ccnn_custom/batch_normalization/beta*
_output_shapes
:	*
dtype0
?
ccnn_custom/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameccnn_custom/conv2d_1/kernel
?
/ccnn_custom/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/conv2d_1/kernel*&
_output_shapes
:	*
dtype0
?
'ccnn_custom/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'ccnn_custom/batch_normalization_1/gamma
?
;ccnn_custom/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp'ccnn_custom/batch_normalization_1/gamma*
_output_shapes
:*
dtype0
?
&ccnn_custom/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&ccnn_custom/batch_normalization_1/beta
?
:ccnn_custom/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp&ccnn_custom/batch_normalization_1/beta*
_output_shapes
:*
dtype0
?
ccnn_custom/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameccnn_custom/conv2d_2/kernel
?
/ccnn_custom/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/conv2d_2/kernel*&
_output_shapes
:*
dtype0
?
'ccnn_custom/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'ccnn_custom/batch_normalization_2/gamma
?
;ccnn_custom/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp'ccnn_custom/batch_normalization_2/gamma*
_output_shapes
:*
dtype0
?
&ccnn_custom/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&ccnn_custom/batch_normalization_2/beta
?
:ccnn_custom/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp&ccnn_custom/batch_normalization_2/beta*
_output_shapes
:*
dtype0
?
ccnn_custom/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameccnn_custom/conv2d_3/kernel
?
/ccnn_custom/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/conv2d_3/kernel*&
_output_shapes
: *
dtype0
?
'ccnn_custom/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'ccnn_custom/batch_normalization_3/gamma
?
;ccnn_custom/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp'ccnn_custom/batch_normalization_3/gamma*
_output_shapes
: *
dtype0
?
&ccnn_custom/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&ccnn_custom/batch_normalization_3/beta
?
:ccnn_custom/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp&ccnn_custom/batch_normalization_3/beta*
_output_shapes
: *
dtype0
?
ccnn_custom/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*,
shared_nameccnn_custom/conv2d_4/kernel
?
/ccnn_custom/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/conv2d_4/kernel*&
_output_shapes
: 0*
dtype0
?
'ccnn_custom/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*8
shared_name)'ccnn_custom/batch_normalization_4/gamma
?
;ccnn_custom/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp'ccnn_custom/batch_normalization_4/gamma*
_output_shapes
:0*
dtype0
?
&ccnn_custom/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*7
shared_name(&ccnn_custom/batch_normalization_4/beta
?
:ccnn_custom/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp&ccnn_custom/batch_normalization_4/beta*
_output_shapes
:0*
dtype0
?
ccnn_custom/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*,
shared_nameccnn_custom/conv2d_5/kernel
?
/ccnn_custom/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpccnn_custom/conv2d_5/kernel*&
_output_shapes
:0@*
dtype0
?
'ccnn_custom/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'ccnn_custom/batch_normalization_5/gamma
?
;ccnn_custom/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp'ccnn_custom/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0
?
&ccnn_custom/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&ccnn_custom/batch_normalization_5/beta
?
:ccnn_custom/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp&ccnn_custom/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
?
+ccnn_custom/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+ccnn_custom/batch_normalization/moving_mean
?
?ccnn_custom/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp+ccnn_custom/batch_normalization/moving_mean*
_output_shapes
:	*
dtype0
?
/ccnn_custom/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*@
shared_name1/ccnn_custom/batch_normalization/moving_variance
?
Cccnn_custom/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp/ccnn_custom/batch_normalization/moving_variance*
_output_shapes
:	*
dtype0
?
-ccnn_custom/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-ccnn_custom/batch_normalization_1/moving_mean
?
Accnn_custom/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp-ccnn_custom/batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
1ccnn_custom/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31ccnn_custom/batch_normalization_1/moving_variance
?
Eccnn_custom/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp1ccnn_custom/batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
?
-ccnn_custom/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-ccnn_custom/batch_normalization_2/moving_mean
?
Accnn_custom/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp-ccnn_custom/batch_normalization_2/moving_mean*
_output_shapes
:*
dtype0
?
1ccnn_custom/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31ccnn_custom/batch_normalization_2/moving_variance
?
Eccnn_custom/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp1ccnn_custom/batch_normalization_2/moving_variance*
_output_shapes
:*
dtype0
?
-ccnn_custom/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-ccnn_custom/batch_normalization_3/moving_mean
?
Accnn_custom/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp-ccnn_custom/batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
?
1ccnn_custom/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31ccnn_custom/batch_normalization_3/moving_variance
?
Eccnn_custom/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp1ccnn_custom/batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
?
-ccnn_custom/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-ccnn_custom/batch_normalization_4/moving_mean
?
Accnn_custom/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp-ccnn_custom/batch_normalization_4/moving_mean*
_output_shapes
:0*
dtype0
?
1ccnn_custom/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*B
shared_name31ccnn_custom/batch_normalization_4/moving_variance
?
Eccnn_custom/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp1ccnn_custom/batch_normalization_4/moving_variance*
_output_shapes
:0*
dtype0
?
-ccnn_custom/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-ccnn_custom/batch_normalization_5/moving_mean
?
Accnn_custom/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp-ccnn_custom/batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
1ccnn_custom/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31ccnn_custom/batch_normalization_5/moving_variance
?
Eccnn_custom/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp1ccnn_custom/batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/ccnn_custom/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*0
shared_name!Adam/ccnn_custom/dense/kernel/m
?
3Adam/ccnn_custom/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ccnn_custom/dense/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/ccnn_custom/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_nameAdam/ccnn_custom/dense/bias/m
?
1Adam/ccnn_custom/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/ccnn_custom/dense/bias/m*
_output_shapes
:
*
dtype0
?
 Adam/ccnn_custom/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/ccnn_custom/conv2d/kernel/m
?
4Adam/ccnn_custom/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp Adam/ccnn_custom/conv2d/kernel/m*&
_output_shapes
:	*
dtype0
?
,Adam/ccnn_custom/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/ccnn_custom/batch_normalization/gamma/m
?
@Adam/ccnn_custom/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp,Adam/ccnn_custom/batch_normalization/gamma/m*
_output_shapes
:	*
dtype0
?
+Adam/ccnn_custom/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+Adam/ccnn_custom/batch_normalization/beta/m
?
?Adam/ccnn_custom/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp+Adam/ccnn_custom/batch_normalization/beta/m*
_output_shapes
:	*
dtype0
?
"Adam/ccnn_custom/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/ccnn_custom/conv2d_1/kernel/m
?
6Adam/ccnn_custom/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_1/kernel/m*&
_output_shapes
:	*
dtype0
?
.Adam/ccnn_custom/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/ccnn_custom/batch_normalization_1/gamma/m
?
BAdam/ccnn_custom/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
-Adam/ccnn_custom/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/ccnn_custom/batch_normalization_1/beta/m
?
AAdam/ccnn_custom/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
"Adam/ccnn_custom/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/ccnn_custom/conv2d_2/kernel/m
?
6Adam/ccnn_custom/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_2/kernel/m*&
_output_shapes
:*
dtype0
?
.Adam/ccnn_custom/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/ccnn_custom/batch_normalization_2/gamma/m
?
BAdam/ccnn_custom/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_2/gamma/m*
_output_shapes
:*
dtype0
?
-Adam/ccnn_custom/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/ccnn_custom/batch_normalization_2/beta/m
?
AAdam/ccnn_custom/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_2/beta/m*
_output_shapes
:*
dtype0
?
"Adam/ccnn_custom/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/ccnn_custom/conv2d_3/kernel/m
?
6Adam/ccnn_custom/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_3/kernel/m*&
_output_shapes
: *
dtype0
?
.Adam/ccnn_custom/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adam/ccnn_custom/batch_normalization_3/gamma/m
?
BAdam/ccnn_custom/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0
?
-Adam/ccnn_custom/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/ccnn_custom/batch_normalization_3/beta/m
?
AAdam/ccnn_custom/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0
?
"Adam/ccnn_custom/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"Adam/ccnn_custom/conv2d_4/kernel/m
?
6Adam/ccnn_custom/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_4/kernel/m*&
_output_shapes
: 0*
dtype0
?
.Adam/ccnn_custom/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.Adam/ccnn_custom/batch_normalization_4/gamma/m
?
BAdam/ccnn_custom/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_4/gamma/m*
_output_shapes
:0*
dtype0
?
-Adam/ccnn_custom/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-Adam/ccnn_custom/batch_normalization_4/beta/m
?
AAdam/ccnn_custom/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_4/beta/m*
_output_shapes
:0*
dtype0
?
"Adam/ccnn_custom/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*3
shared_name$"Adam/ccnn_custom/conv2d_5/kernel/m
?
6Adam/ccnn_custom/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_5/kernel/m*&
_output_shapes
:0@*
dtype0
?
.Adam/ccnn_custom/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/ccnn_custom/batch_normalization_5/gamma/m
?
BAdam/ccnn_custom/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_5/gamma/m*
_output_shapes
:@*
dtype0
?
-Adam/ccnn_custom/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/ccnn_custom/batch_normalization_5/beta/m
?
AAdam/ccnn_custom/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_5/beta/m*
_output_shapes
:@*
dtype0
?
Adam/ccnn_custom/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*0
shared_name!Adam/ccnn_custom/dense/kernel/v
?
3Adam/ccnn_custom/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ccnn_custom/dense/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/ccnn_custom/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_nameAdam/ccnn_custom/dense/bias/v
?
1Adam/ccnn_custom/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/ccnn_custom/dense/bias/v*
_output_shapes
:
*
dtype0
?
 Adam/ccnn_custom/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/ccnn_custom/conv2d/kernel/v
?
4Adam/ccnn_custom/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp Adam/ccnn_custom/conv2d/kernel/v*&
_output_shapes
:	*
dtype0
?
,Adam/ccnn_custom/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*=
shared_name.,Adam/ccnn_custom/batch_normalization/gamma/v
?
@Adam/ccnn_custom/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp,Adam/ccnn_custom/batch_normalization/gamma/v*
_output_shapes
:	*
dtype0
?
+Adam/ccnn_custom/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+Adam/ccnn_custom/batch_normalization/beta/v
?
?Adam/ccnn_custom/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp+Adam/ccnn_custom/batch_normalization/beta/v*
_output_shapes
:	*
dtype0
?
"Adam/ccnn_custom/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"Adam/ccnn_custom/conv2d_1/kernel/v
?
6Adam/ccnn_custom/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_1/kernel/v*&
_output_shapes
:	*
dtype0
?
.Adam/ccnn_custom/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/ccnn_custom/batch_normalization_1/gamma/v
?
BAdam/ccnn_custom/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
-Adam/ccnn_custom/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/ccnn_custom/batch_normalization_1/beta/v
?
AAdam/ccnn_custom/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
"Adam/ccnn_custom/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/ccnn_custom/conv2d_2/kernel/v
?
6Adam/ccnn_custom/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_2/kernel/v*&
_output_shapes
:*
dtype0
?
.Adam/ccnn_custom/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/ccnn_custom/batch_normalization_2/gamma/v
?
BAdam/ccnn_custom/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_2/gamma/v*
_output_shapes
:*
dtype0
?
-Adam/ccnn_custom/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/ccnn_custom/batch_normalization_2/beta/v
?
AAdam/ccnn_custom/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_2/beta/v*
_output_shapes
:*
dtype0
?
"Adam/ccnn_custom/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/ccnn_custom/conv2d_3/kernel/v
?
6Adam/ccnn_custom/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_3/kernel/v*&
_output_shapes
: *
dtype0
?
.Adam/ccnn_custom/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adam/ccnn_custom/batch_normalization_3/gamma/v
?
BAdam/ccnn_custom/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0
?
-Adam/ccnn_custom/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/ccnn_custom/batch_normalization_3/beta/v
?
AAdam/ccnn_custom/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0
?
"Adam/ccnn_custom/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*3
shared_name$"Adam/ccnn_custom/conv2d_4/kernel/v
?
6Adam/ccnn_custom/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_4/kernel/v*&
_output_shapes
: 0*
dtype0
?
.Adam/ccnn_custom/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*?
shared_name0.Adam/ccnn_custom/batch_normalization_4/gamma/v
?
BAdam/ccnn_custom/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_4/gamma/v*
_output_shapes
:0*
dtype0
?
-Adam/ccnn_custom/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*>
shared_name/-Adam/ccnn_custom/batch_normalization_4/beta/v
?
AAdam/ccnn_custom/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_4/beta/v*
_output_shapes
:0*
dtype0
?
"Adam/ccnn_custom/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*3
shared_name$"Adam/ccnn_custom/conv2d_5/kernel/v
?
6Adam/ccnn_custom/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/ccnn_custom/conv2d_5/kernel/v*&
_output_shapes
:0@*
dtype0
?
.Adam/ccnn_custom/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/ccnn_custom/batch_normalization_5/gamma/v
?
BAdam/ccnn_custom/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/ccnn_custom/batch_normalization_5/gamma/v*
_output_shapes
:@*
dtype0
?
-Adam/ccnn_custom/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/ccnn_custom/batch_normalization_5/beta/v
?
AAdam/ccnn_custom/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp-Adam/ccnn_custom/batch_normalization_5/beta/v*
_output_shapes
:@*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?

Config

InputShape
ConvLayerFeatures
ConvWindows
PoolWindows
KerasLayers
FlatteningLayer
OutputLayer
	SoftmaxActivation

	optimizer
	Structure
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
\
CNN.InputShape
CNN.ConvOutputFeatures
CNN.ConvWindows
CNN.PoolWindows* 
* 
* 
,
0
1
2
3
4
5* 
	
4* 
?
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18*
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
?

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
?
Biter

Cbeta_1

Dbeta_2
	Edecay
Flearning_rate4m?5m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?4v?5v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?*
?
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
W16
X17
Y18
Z19
[20
\21* 
?
]0
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12
j13
k14
l15
m16
n17
o18
p19
q20
r21
s22
t23
u24
v25
w26
x27
y28
z29
430
531*
?
]0
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12
j13
k14
l15
m16
n17
418
519*
-
{0
|1
}2
~3
4
?5* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

?serving_default* 
* 
* 
* 
* 
* 
* 
* 
?

]kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	^gamma
_beta
omoving_mean
pmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

`kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	agamma
bbeta
qmoving_mean
rmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

ckernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	dgamma
ebeta
smoving_mean
tmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

fkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	ggamma
hbeta
umoving_mean
vmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

ikernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	jgamma
kbeta
wmoving_mean
xmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

lkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
	?axis
	mgamma
nbeta
ymoving_mean
zmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEccnn_custom/dense/kernel-OutputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEccnn_custom/dense/bias+OutputLayer/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
YS
VARIABLE_VALUEccnn_custom/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%ccnn_custom/batch_normalization/gamma&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$ccnn_custom/batch_normalization/beta&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEccnn_custom/conv2d_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'ccnn_custom/batch_normalization_1/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&ccnn_custom/batch_normalization_1/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEccnn_custom/conv2d_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'ccnn_custom/batch_normalization_2/gamma&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&ccnn_custom/batch_normalization_2/beta&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEccnn_custom/conv2d_3/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'ccnn_custom/batch_normalization_3/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&ccnn_custom/batch_normalization_3/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEccnn_custom/conv2d_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'ccnn_custom/batch_normalization_4/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&ccnn_custom/batch_normalization_4/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEccnn_custom/conv2d_5/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'ccnn_custom/batch_normalization_5/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&ccnn_custom/batch_normalization_5/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+ccnn_custom/batch_normalization/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/ccnn_custom/batch_normalization/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-ccnn_custom/batch_normalization_1/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1ccnn_custom/batch_normalization_1/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-ccnn_custom/batch_normalization_2/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1ccnn_custom/batch_normalization_2/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-ccnn_custom/batch_normalization_3/moving_mean'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1ccnn_custom/batch_normalization_3/moving_variance'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-ccnn_custom/batch_normalization_4/moving_mean'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1ccnn_custom/batch_normalization_4/moving_variance'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-ccnn_custom/batch_normalization_5/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1ccnn_custom/batch_normalization_5/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
Z
o0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11*
?
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18
19
20
	21*

?0
?1*
* 
* 
* 

]0*

]0*
	
{0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
^0
_1
o2
p3*

^0
_1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

`0*

`0*
	
|0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
a0
b1
q2
r3*

a0
b1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

c0*

c0*
	
}0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
d0
e1
s2
t3*

d0
e1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

f0*

f0*
	
~0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
g0
h1
u2
v3*

g0
h1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

i0*

i0*
	
0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
j0
k1
w2
x3*

j0
k1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

l0*

l0*


?0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
 
m0
n1
y2
z3*

m0
n1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
* 
* 
* 
	
{0* 
* 
* 
* 
* 
* 
* 

o0
p1*
* 
* 
* 
* 
* 
* 
* 
	
|0* 
* 
* 
* 
* 
* 
* 

q0
r1*
* 
* 
* 
* 
* 
* 
* 
	
}0* 
* 
* 
* 
* 
* 
* 

s0
t1*
* 
* 
* 
* 
* 
* 
* 
	
~0* 
* 
* 
* 
* 
* 
* 

u0
v1*
* 
* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 

w0
x1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


?0* 
* 
* 
* 
* 
* 
* 

y0
z1*
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
?|
VARIABLE_VALUEAdam/ccnn_custom/dense/kernel/mIOutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/ccnn_custom/dense/bias/mGOutputLayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/ccnn_custom/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/ccnn_custom/batch_normalization/gamma/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/ccnn_custom/batch_normalization/beta/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_1/gamma/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_1/beta/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_2/gamma/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_2/beta/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_3/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_3/gamma/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_3/beta/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_4/gamma/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_4/beta/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_5/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_5/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_5/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/ccnn_custom/dense/kernel/vIOutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/ccnn_custom/dense/bias/vGOutputLayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/ccnn_custom/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/ccnn_custom/batch_normalization/gamma/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/ccnn_custom/batch_normalization/beta/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_1/gamma/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_1/beta/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_2/gamma/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_2/beta/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_3/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_3/gamma/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_3/beta/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_4/gamma/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_4/beta/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE"Adam/ccnn_custom/conv2d_5/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/ccnn_custom/batch_normalization_5/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adam/ccnn_custom/batch_normalization_5/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ccnn_custom/conv2d/kernel%ccnn_custom/batch_normalization/gamma$ccnn_custom/batch_normalization/beta+ccnn_custom/batch_normalization/moving_mean/ccnn_custom/batch_normalization/moving_varianceccnn_custom/conv2d_1/kernel'ccnn_custom/batch_normalization_1/gamma&ccnn_custom/batch_normalization_1/beta-ccnn_custom/batch_normalization_1/moving_mean1ccnn_custom/batch_normalization_1/moving_varianceccnn_custom/conv2d_2/kernel'ccnn_custom/batch_normalization_2/gamma&ccnn_custom/batch_normalization_2/beta-ccnn_custom/batch_normalization_2/moving_mean1ccnn_custom/batch_normalization_2/moving_varianceccnn_custom/conv2d_3/kernel'ccnn_custom/batch_normalization_3/gamma&ccnn_custom/batch_normalization_3/beta-ccnn_custom/batch_normalization_3/moving_mean1ccnn_custom/batch_normalization_3/moving_varianceccnn_custom/conv2d_4/kernel'ccnn_custom/batch_normalization_4/gamma&ccnn_custom/batch_normalization_4/beta-ccnn_custom/batch_normalization_4/moving_mean1ccnn_custom/batch_normalization_4/moving_varianceccnn_custom/conv2d_5/kernel'ccnn_custom/batch_normalization_5/gamma&ccnn_custom/batch_normalization_5/beta-ccnn_custom/batch_normalization_5/moving_mean1ccnn_custom/batch_normalization_5/moving_varianceccnn_custom/dense/kernelccnn_custom/dense/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_10296
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,ccnn_custom/dense/kernel/Read/ReadVariableOp*ccnn_custom/dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-ccnn_custom/conv2d/kernel/Read/ReadVariableOp9ccnn_custom/batch_normalization/gamma/Read/ReadVariableOp8ccnn_custom/batch_normalization/beta/Read/ReadVariableOp/ccnn_custom/conv2d_1/kernel/Read/ReadVariableOp;ccnn_custom/batch_normalization_1/gamma/Read/ReadVariableOp:ccnn_custom/batch_normalization_1/beta/Read/ReadVariableOp/ccnn_custom/conv2d_2/kernel/Read/ReadVariableOp;ccnn_custom/batch_normalization_2/gamma/Read/ReadVariableOp:ccnn_custom/batch_normalization_2/beta/Read/ReadVariableOp/ccnn_custom/conv2d_3/kernel/Read/ReadVariableOp;ccnn_custom/batch_normalization_3/gamma/Read/ReadVariableOp:ccnn_custom/batch_normalization_3/beta/Read/ReadVariableOp/ccnn_custom/conv2d_4/kernel/Read/ReadVariableOp;ccnn_custom/batch_normalization_4/gamma/Read/ReadVariableOp:ccnn_custom/batch_normalization_4/beta/Read/ReadVariableOp/ccnn_custom/conv2d_5/kernel/Read/ReadVariableOp;ccnn_custom/batch_normalization_5/gamma/Read/ReadVariableOp:ccnn_custom/batch_normalization_5/beta/Read/ReadVariableOp?ccnn_custom/batch_normalization/moving_mean/Read/ReadVariableOpCccnn_custom/batch_normalization/moving_variance/Read/ReadVariableOpAccnn_custom/batch_normalization_1/moving_mean/Read/ReadVariableOpEccnn_custom/batch_normalization_1/moving_variance/Read/ReadVariableOpAccnn_custom/batch_normalization_2/moving_mean/Read/ReadVariableOpEccnn_custom/batch_normalization_2/moving_variance/Read/ReadVariableOpAccnn_custom/batch_normalization_3/moving_mean/Read/ReadVariableOpEccnn_custom/batch_normalization_3/moving_variance/Read/ReadVariableOpAccnn_custom/batch_normalization_4/moving_mean/Read/ReadVariableOpEccnn_custom/batch_normalization_4/moving_variance/Read/ReadVariableOpAccnn_custom/batch_normalization_5/moving_mean/Read/ReadVariableOpEccnn_custom/batch_normalization_5/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/ccnn_custom/dense/kernel/m/Read/ReadVariableOp1Adam/ccnn_custom/dense/bias/m/Read/ReadVariableOp4Adam/ccnn_custom/conv2d/kernel/m/Read/ReadVariableOp@Adam/ccnn_custom/batch_normalization/gamma/m/Read/ReadVariableOp?Adam/ccnn_custom/batch_normalization/beta/m/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_1/kernel/m/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_1/gamma/m/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_1/beta/m/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_2/kernel/m/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_2/gamma/m/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_2/beta/m/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_3/kernel/m/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_3/gamma/m/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_3/beta/m/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_4/kernel/m/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_4/gamma/m/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_4/beta/m/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_5/kernel/m/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_5/gamma/m/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_5/beta/m/Read/ReadVariableOp3Adam/ccnn_custom/dense/kernel/v/Read/ReadVariableOp1Adam/ccnn_custom/dense/bias/v/Read/ReadVariableOp4Adam/ccnn_custom/conv2d/kernel/v/Read/ReadVariableOp@Adam/ccnn_custom/batch_normalization/gamma/v/Read/ReadVariableOp?Adam/ccnn_custom/batch_normalization/beta/v/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_1/kernel/v/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_1/gamma/v/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_1/beta/v/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_2/kernel/v/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_2/gamma/v/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_2/beta/v/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_3/kernel/v/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_3/gamma/v/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_3/beta/v/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_4/kernel/v/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_4/gamma/v/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_4/beta/v/Read/ReadVariableOp6Adam/ccnn_custom/conv2d_5/kernel/v/Read/ReadVariableOpBAdam/ccnn_custom/batch_normalization_5/gamma/v/Read/ReadVariableOpAAdam/ccnn_custom/batch_normalization_5/beta/v/Read/ReadVariableOpConst*^
TinW
U2S	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_11266
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameccnn_custom/dense/kernelccnn_custom/dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateccnn_custom/conv2d/kernel%ccnn_custom/batch_normalization/gamma$ccnn_custom/batch_normalization/betaccnn_custom/conv2d_1/kernel'ccnn_custom/batch_normalization_1/gamma&ccnn_custom/batch_normalization_1/betaccnn_custom/conv2d_2/kernel'ccnn_custom/batch_normalization_2/gamma&ccnn_custom/batch_normalization_2/betaccnn_custom/conv2d_3/kernel'ccnn_custom/batch_normalization_3/gamma&ccnn_custom/batch_normalization_3/betaccnn_custom/conv2d_4/kernel'ccnn_custom/batch_normalization_4/gamma&ccnn_custom/batch_normalization_4/betaccnn_custom/conv2d_5/kernel'ccnn_custom/batch_normalization_5/gamma&ccnn_custom/batch_normalization_5/beta+ccnn_custom/batch_normalization/moving_mean/ccnn_custom/batch_normalization/moving_variance-ccnn_custom/batch_normalization_1/moving_mean1ccnn_custom/batch_normalization_1/moving_variance-ccnn_custom/batch_normalization_2/moving_mean1ccnn_custom/batch_normalization_2/moving_variance-ccnn_custom/batch_normalization_3/moving_mean1ccnn_custom/batch_normalization_3/moving_variance-ccnn_custom/batch_normalization_4/moving_mean1ccnn_custom/batch_normalization_4/moving_variance-ccnn_custom/batch_normalization_5/moving_mean1ccnn_custom/batch_normalization_5/moving_variancetotalcounttotal_1count_1Adam/ccnn_custom/dense/kernel/mAdam/ccnn_custom/dense/bias/m Adam/ccnn_custom/conv2d/kernel/m,Adam/ccnn_custom/batch_normalization/gamma/m+Adam/ccnn_custom/batch_normalization/beta/m"Adam/ccnn_custom/conv2d_1/kernel/m.Adam/ccnn_custom/batch_normalization_1/gamma/m-Adam/ccnn_custom/batch_normalization_1/beta/m"Adam/ccnn_custom/conv2d_2/kernel/m.Adam/ccnn_custom/batch_normalization_2/gamma/m-Adam/ccnn_custom/batch_normalization_2/beta/m"Adam/ccnn_custom/conv2d_3/kernel/m.Adam/ccnn_custom/batch_normalization_3/gamma/m-Adam/ccnn_custom/batch_normalization_3/beta/m"Adam/ccnn_custom/conv2d_4/kernel/m.Adam/ccnn_custom/batch_normalization_4/gamma/m-Adam/ccnn_custom/batch_normalization_4/beta/m"Adam/ccnn_custom/conv2d_5/kernel/m.Adam/ccnn_custom/batch_normalization_5/gamma/m-Adam/ccnn_custom/batch_normalization_5/beta/mAdam/ccnn_custom/dense/kernel/vAdam/ccnn_custom/dense/bias/v Adam/ccnn_custom/conv2d/kernel/v,Adam/ccnn_custom/batch_normalization/gamma/v+Adam/ccnn_custom/batch_normalization/beta/v"Adam/ccnn_custom/conv2d_1/kernel/v.Adam/ccnn_custom/batch_normalization_1/gamma/v-Adam/ccnn_custom/batch_normalization_1/beta/v"Adam/ccnn_custom/conv2d_2/kernel/v.Adam/ccnn_custom/batch_normalization_2/gamma/v-Adam/ccnn_custom/batch_normalization_2/beta/v"Adam/ccnn_custom/conv2d_3/kernel/v.Adam/ccnn_custom/batch_normalization_3/gamma/v-Adam/ccnn_custom/batch_normalization_3/beta/v"Adam/ccnn_custom/conv2d_4/kernel/v.Adam/ccnn_custom/batch_normalization_4/gamma/v-Adam/ccnn_custom/batch_normalization_4/beta/v"Adam/ccnn_custom/conv2d_5/kernel/v.Adam/ccnn_custom/batch_normalization_5/gamma/v-Adam/ccnn_custom/batch_normalization_5/beta/v*]
TinV
T2R*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_11519??
?
^
B__inference_softmax_layer_call_and_return_conditional_losses_10336

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10892

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8418

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_activation_5_layer_call_and_return_conditional_losses_8892

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
H
,__inference_activation_3_layer_call_fn_10727

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_8825h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_8883

inputs8
conv2d_readvariableop_resource:0@
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????0: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
ޟ
?
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9603
input_1%
conv2d_9480:	&
batch_normalization_9484:	&
batch_normalization_9486:	&
batch_normalization_9488:	&
batch_normalization_9490:	'
conv2d_1_9493:	(
batch_normalization_1_9497:(
batch_normalization_1_9499:(
batch_normalization_1_9501:(
batch_normalization_1_9503:'
conv2d_2_9506:(
batch_normalization_2_9510:(
batch_normalization_2_9512:(
batch_normalization_2_9514:(
batch_normalization_2_9516:'
conv2d_3_9519: (
batch_normalization_3_9523: (
batch_normalization_3_9525: (
batch_normalization_3_9527: (
batch_normalization_3_9529: '
conv2d_4_9532: 0(
batch_normalization_4_9536:0(
batch_normalization_4_9538:0(
batch_normalization_4_9540:0(
batch_normalization_4_9542:0'
conv2d_5_9546:0@(
batch_normalization_5_9550:@(
batch_normalization_5_9552:@(
batch_normalization_5_9554:@(
batch_normalization_5_9556:@

dense_9560:	?


dense_9562:

identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_9480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_8717?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_8726?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_9484batch_normalization_9486batch_normalization_9488batch_normalization_9490*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8323?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_9493*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8750?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_8759?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_1_9497batch_normalization_1_9499batch_normalization_1_9501batch_normalization_1_9503*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8387?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_9506*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8783?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_8792?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_2_9510batch_normalization_2_9512batch_normalization_2_9514batch_normalization_2_9516*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8451?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_9519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8816?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_8825?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_3_9523batch_normalization_3_9525batch_normalization_3_9527batch_normalization_3_9529*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8515?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_9532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8849?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_8858?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_4_9536batch_normalization_4_9538batch_normalization_4_9540batch_normalization_4_9542*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8579?
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8630?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_5_9546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_8883?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_8892?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_5_9550batch_normalization_5_9552batch_normalization_5_9554batch_normalization_5_9556*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8655?
flatten/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8909?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_9560
dense_9562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8921?
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_softmax_layer_call_and_return_conditional_losses_8932?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9480*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_9493*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_9506*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_9519*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_9532*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_9546*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10982

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_10938

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_10369`
Fccnn_custom_conv2d_2_kernel_regularizer_square_readvariableop_resource:
identity??=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFccnn_custom_conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/ccnn_custom/conv2d_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_3_10380`
Fccnn_custom_conv2d_3_kernel_regularizer_square_readvariableop_resource: 
identity??=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFccnn_custom_conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/ccnn_custom/conv2d_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp
՟
?
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9341
p_tinput%
conv2d_9218:	&
batch_normalization_9222:	&
batch_normalization_9224:	&
batch_normalization_9226:	&
batch_normalization_9228:	'
conv2d_1_9231:	(
batch_normalization_1_9235:(
batch_normalization_1_9237:(
batch_normalization_1_9239:(
batch_normalization_1_9241:'
conv2d_2_9244:(
batch_normalization_2_9248:(
batch_normalization_2_9250:(
batch_normalization_2_9252:(
batch_normalization_2_9254:'
conv2d_3_9257: (
batch_normalization_3_9261: (
batch_normalization_3_9263: (
batch_normalization_3_9265: (
batch_normalization_3_9267: '
conv2d_4_9270: 0(
batch_normalization_4_9274:0(
batch_normalization_4_9276:0(
batch_normalization_4_9278:0(
batch_normalization_4_9280:0'
conv2d_5_9284:0@(
batch_normalization_5_9288:@(
batch_normalization_5_9290:@(
batch_normalization_5_9292:@(
batch_normalization_5_9294:@

dense_9298:	?


dense_9300:

identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallp_tinputconv2d_9218*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_8717?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_8726?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_9222batch_normalization_9224batch_normalization_9226batch_normalization_9228*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8354?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_9231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8750?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_8759?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_1_9235batch_normalization_1_9237batch_normalization_1_9239batch_normalization_1_9241*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8418?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_9244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8783?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_8792?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_2_9248batch_normalization_2_9250batch_normalization_2_9252batch_normalization_2_9254*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8482?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_9257*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8816?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_8825?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_3_9261batch_normalization_3_9263batch_normalization_3_9265batch_normalization_3_9267*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8546?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_9270*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8849?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_8858?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_4_9274batch_normalization_4_9276batch_normalization_4_9278batch_normalization_4_9280*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8610?
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8630?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_5_9284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_8883?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_8892?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_5_9288batch_normalization_5_9290batch_normalization_5_9292batch_normalization_5_9294*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8686?
flatten/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8909?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_9298
dense_9300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8921?
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_softmax_layer_call_and_return_conditional_losses_8932?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9218*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_9231*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_9244*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_9257*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_9270*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_9284*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
p_tInput
?
a
E__inference_activation_layer_call_and_return_conditional_losses_10438

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????	:W S
/
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10482

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_10897

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8630?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8655

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
F__inference_ccnn_custom_layer_call_and_return_conditional_losses_10067
p_tinput?
%conv2d_conv2d_readvariableop_resource:	9
+batch_normalization_readvariableop_resource:	;
-batch_normalization_readvariableop_1_resource:	J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	A
'conv2d_1_conv2d_readvariableop_resource:	;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_3_conv2d_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_4_conv2d_readvariableop_resource: 0;
-batch_normalization_4_readvariableop_resource:0=
/batch_normalization_4_readvariableop_1_resource:0L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_5_conv2d_readvariableop_resource:0@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@7
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
conv2d/Conv2DConv2Dp_tinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingSAME*
strides
i
activation/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:?????????	?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:	*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????	:	:	:	:	:*
epsilon%o?:*
is_training( ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
m
activation_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:??????????
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
m
activation_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:??????????
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
m
activation_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:????????? ?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
m
activation_4/ReluReluconv2d_4/Conv2D:output:0*
T0*/
_output_shapes
:?????????0?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:0*
dtype0?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
is_training( ?
max_pooling2d/MaxPoolMaxPool*batch_normalization_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
conv2d_5/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
m
activation_5/ReluReluconv2d_5/Conv2D:output:0*
T0*/
_output_shapes
:?????????@?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten/ReshapeReshape*batch_normalization_5/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
p_tInput
?
?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10928

inputs8
conv2d_readvariableop_resource:0@
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????0: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
F
*__inference_activation_layer_call_fn_10433

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_8726h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????	:W S
/
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8816

inputs8
conv2d_readvariableop_resource: 
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_2_layer_call_fn_10611

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8783w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
§
?,
__inference__traced_save_11266
file_prefix7
3savev2_ccnn_custom_dense_kernel_read_readvariableop5
1savev2_ccnn_custom_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_ccnn_custom_conv2d_kernel_read_readvariableopD
@savev2_ccnn_custom_batch_normalization_gamma_read_readvariableopC
?savev2_ccnn_custom_batch_normalization_beta_read_readvariableop:
6savev2_ccnn_custom_conv2d_1_kernel_read_readvariableopF
Bsavev2_ccnn_custom_batch_normalization_1_gamma_read_readvariableopE
Asavev2_ccnn_custom_batch_normalization_1_beta_read_readvariableop:
6savev2_ccnn_custom_conv2d_2_kernel_read_readvariableopF
Bsavev2_ccnn_custom_batch_normalization_2_gamma_read_readvariableopE
Asavev2_ccnn_custom_batch_normalization_2_beta_read_readvariableop:
6savev2_ccnn_custom_conv2d_3_kernel_read_readvariableopF
Bsavev2_ccnn_custom_batch_normalization_3_gamma_read_readvariableopE
Asavev2_ccnn_custom_batch_normalization_3_beta_read_readvariableop:
6savev2_ccnn_custom_conv2d_4_kernel_read_readvariableopF
Bsavev2_ccnn_custom_batch_normalization_4_gamma_read_readvariableopE
Asavev2_ccnn_custom_batch_normalization_4_beta_read_readvariableop:
6savev2_ccnn_custom_conv2d_5_kernel_read_readvariableopF
Bsavev2_ccnn_custom_batch_normalization_5_gamma_read_readvariableopE
Asavev2_ccnn_custom_batch_normalization_5_beta_read_readvariableopJ
Fsavev2_ccnn_custom_batch_normalization_moving_mean_read_readvariableopN
Jsavev2_ccnn_custom_batch_normalization_moving_variance_read_readvariableopL
Hsavev2_ccnn_custom_batch_normalization_1_moving_mean_read_readvariableopP
Lsavev2_ccnn_custom_batch_normalization_1_moving_variance_read_readvariableopL
Hsavev2_ccnn_custom_batch_normalization_2_moving_mean_read_readvariableopP
Lsavev2_ccnn_custom_batch_normalization_2_moving_variance_read_readvariableopL
Hsavev2_ccnn_custom_batch_normalization_3_moving_mean_read_readvariableopP
Lsavev2_ccnn_custom_batch_normalization_3_moving_variance_read_readvariableopL
Hsavev2_ccnn_custom_batch_normalization_4_moving_mean_read_readvariableopP
Lsavev2_ccnn_custom_batch_normalization_4_moving_variance_read_readvariableopL
Hsavev2_ccnn_custom_batch_normalization_5_moving_mean_read_readvariableopP
Lsavev2_ccnn_custom_batch_normalization_5_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_adam_ccnn_custom_dense_kernel_m_read_readvariableop<
8savev2_adam_ccnn_custom_dense_bias_m_read_readvariableop?
;savev2_adam_ccnn_custom_conv2d_kernel_m_read_readvariableopK
Gsavev2_adam_ccnn_custom_batch_normalization_gamma_m_read_readvariableopJ
Fsavev2_adam_ccnn_custom_batch_normalization_beta_m_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_1_kernel_m_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_1_gamma_m_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_1_beta_m_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_2_kernel_m_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_2_gamma_m_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_2_beta_m_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_3_kernel_m_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_3_gamma_m_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_3_beta_m_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_4_kernel_m_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_4_gamma_m_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_4_beta_m_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_5_kernel_m_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_5_gamma_m_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_5_beta_m_read_readvariableop>
:savev2_adam_ccnn_custom_dense_kernel_v_read_readvariableop<
8savev2_adam_ccnn_custom_dense_bias_v_read_readvariableop?
;savev2_adam_ccnn_custom_conv2d_kernel_v_read_readvariableopK
Gsavev2_adam_ccnn_custom_batch_normalization_gamma_v_read_readvariableopJ
Fsavev2_adam_ccnn_custom_batch_normalization_beta_v_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_1_kernel_v_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_1_gamma_v_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_1_beta_v_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_2_kernel_v_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_2_gamma_v_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_2_beta_v_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_3_kernel_v_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_3_gamma_v_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_3_beta_v_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_4_kernel_v_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_4_gamma_v_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_4_beta_v_read_readvariableopA
=savev2_adam_ccnn_custom_conv2d_5_kernel_v_read_readvariableopM
Isavev2_adam_ccnn_custom_batch_normalization_5_gamma_v_read_readvariableopL
Hsavev2_adam_ccnn_custom_batch_normalization_5_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?#
value?#B?#RB-OutputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+OutputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIOutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGOutputLayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIOutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGOutputLayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?
value?B?RB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_ccnn_custom_dense_kernel_read_readvariableop1savev2_ccnn_custom_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_ccnn_custom_conv2d_kernel_read_readvariableop@savev2_ccnn_custom_batch_normalization_gamma_read_readvariableop?savev2_ccnn_custom_batch_normalization_beta_read_readvariableop6savev2_ccnn_custom_conv2d_1_kernel_read_readvariableopBsavev2_ccnn_custom_batch_normalization_1_gamma_read_readvariableopAsavev2_ccnn_custom_batch_normalization_1_beta_read_readvariableop6savev2_ccnn_custom_conv2d_2_kernel_read_readvariableopBsavev2_ccnn_custom_batch_normalization_2_gamma_read_readvariableopAsavev2_ccnn_custom_batch_normalization_2_beta_read_readvariableop6savev2_ccnn_custom_conv2d_3_kernel_read_readvariableopBsavev2_ccnn_custom_batch_normalization_3_gamma_read_readvariableopAsavev2_ccnn_custom_batch_normalization_3_beta_read_readvariableop6savev2_ccnn_custom_conv2d_4_kernel_read_readvariableopBsavev2_ccnn_custom_batch_normalization_4_gamma_read_readvariableopAsavev2_ccnn_custom_batch_normalization_4_beta_read_readvariableop6savev2_ccnn_custom_conv2d_5_kernel_read_readvariableopBsavev2_ccnn_custom_batch_normalization_5_gamma_read_readvariableopAsavev2_ccnn_custom_batch_normalization_5_beta_read_readvariableopFsavev2_ccnn_custom_batch_normalization_moving_mean_read_readvariableopJsavev2_ccnn_custom_batch_normalization_moving_variance_read_readvariableopHsavev2_ccnn_custom_batch_normalization_1_moving_mean_read_readvariableopLsavev2_ccnn_custom_batch_normalization_1_moving_variance_read_readvariableopHsavev2_ccnn_custom_batch_normalization_2_moving_mean_read_readvariableopLsavev2_ccnn_custom_batch_normalization_2_moving_variance_read_readvariableopHsavev2_ccnn_custom_batch_normalization_3_moving_mean_read_readvariableopLsavev2_ccnn_custom_batch_normalization_3_moving_variance_read_readvariableopHsavev2_ccnn_custom_batch_normalization_4_moving_mean_read_readvariableopLsavev2_ccnn_custom_batch_normalization_4_moving_variance_read_readvariableopHsavev2_ccnn_custom_batch_normalization_5_moving_mean_read_readvariableopLsavev2_ccnn_custom_batch_normalization_5_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_ccnn_custom_dense_kernel_m_read_readvariableop8savev2_adam_ccnn_custom_dense_bias_m_read_readvariableop;savev2_adam_ccnn_custom_conv2d_kernel_m_read_readvariableopGsavev2_adam_ccnn_custom_batch_normalization_gamma_m_read_readvariableopFsavev2_adam_ccnn_custom_batch_normalization_beta_m_read_readvariableop=savev2_adam_ccnn_custom_conv2d_1_kernel_m_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_1_gamma_m_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_1_beta_m_read_readvariableop=savev2_adam_ccnn_custom_conv2d_2_kernel_m_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_2_gamma_m_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_2_beta_m_read_readvariableop=savev2_adam_ccnn_custom_conv2d_3_kernel_m_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_3_gamma_m_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_3_beta_m_read_readvariableop=savev2_adam_ccnn_custom_conv2d_4_kernel_m_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_4_gamma_m_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_4_beta_m_read_readvariableop=savev2_adam_ccnn_custom_conv2d_5_kernel_m_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_5_gamma_m_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_5_beta_m_read_readvariableop:savev2_adam_ccnn_custom_dense_kernel_v_read_readvariableop8savev2_adam_ccnn_custom_dense_bias_v_read_readvariableop;savev2_adam_ccnn_custom_conv2d_kernel_v_read_readvariableopGsavev2_adam_ccnn_custom_batch_normalization_gamma_v_read_readvariableopFsavev2_adam_ccnn_custom_batch_normalization_beta_v_read_readvariableop=savev2_adam_ccnn_custom_conv2d_1_kernel_v_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_1_gamma_v_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_1_beta_v_read_readvariableop=savev2_adam_ccnn_custom_conv2d_2_kernel_v_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_2_gamma_v_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_2_beta_v_read_readvariableop=savev2_adam_ccnn_custom_conv2d_3_kernel_v_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_3_gamma_v_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_3_beta_v_read_readvariableop=savev2_adam_ccnn_custom_conv2d_4_kernel_v_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_4_gamma_v_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_4_beta_v_read_readvariableop=savev2_adam_ccnn_custom_conv2d_5_kernel_v_read_readvariableopIsavev2_adam_ccnn_custom_batch_normalization_5_gamma_v_read_readvariableopHsavev2_adam_ccnn_custom_batch_normalization_5_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *`
dtypesV
T2R	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?
:
: : : : : :	:	:	:	:::::: : : : 0:0:0:0@:@:@:	:	::::: : :0:0:@:@: : : : :	?
:
:	:	:	:	:::::: : : : 0:0:0:0@:@:@:	?
:
:	:	:	:	:::::: : : : 0:0:0:0@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:	: 	

_output_shapes
:	: 


_output_shapes
:	:,(
&
_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:	: 

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
:0: #

_output_shapes
:0: $

_output_shapes
:@: %

_output_shapes
:@:&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :%*!

_output_shapes
:	?
: +

_output_shapes
:
:,,(
&
_output_shapes
:	: -

_output_shapes
:	: .

_output_shapes
:	:,/(
&
_output_shapes
:	: 0

_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
::,5(
&
_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: 0: 9

_output_shapes
:0: :

_output_shapes
:0:,;(
&
_output_shapes
:0@: <

_output_shapes
:@: =

_output_shapes
:@:%>!

_output_shapes
:	?
: ?

_output_shapes
:
:,@(
&
_output_shapes
:	: A

_output_shapes
:	: B

_output_shapes
:	:,C(
&
_output_shapes
:	: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
:: H

_output_shapes
::,I(
&
_output_shapes
: : J

_output_shapes
: : K

_output_shapes
: :,L(
&
_output_shapes
: 0: M

_output_shapes
:0: N

_output_shapes
:0:,O(
&
_output_shapes
:0@: P

_output_shapes
:@: Q

_output_shapes
:@:R

_output_shapes
: 
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10598

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10722

inputs8
conv2d_readvariableop_resource: 
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_3_layer_call_fn_10758

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8546?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_10358`
Fccnn_custom_conv2d_1_kernel_regularizer_square_readvariableop_resource:	
identity??=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFccnn_custom_conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/ccnn_custom/conv2d_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp
?
?
*__inference_ccnn_custom_layer_call_fn_9909
p_tinput!
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	#
	unknown_4:	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:$

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 0

unknown_20:0

unknown_21:0

unknown_22:0

unknown_23:0$

unknown_24:0@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	?


unknown_30:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallp_tinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
p_tInput
?
?
%__inference_dense_layer_call_fn_10316

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10874

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_5_layer_call_fn_10951

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8655?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_ccnn_custom_layer_call_fn_9038
input_1!
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	#
	unknown_4:	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:$

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 0

unknown_20:0

unknown_21:0

unknown_22:0

unknown_23:0$

unknown_24:0@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	?


unknown_30:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_8971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
]
A__inference_softmax_layer_call_and_return_conditional_losses_8932

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_2_layer_call_fn_10647

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8451?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_10536

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_4_layer_call_fn_10843

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8579?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_10428

inputs8
conv2d_readvariableop_resource:	
identity??Conv2D/ReadVariableOp?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingSAME*
strides
?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????	?
NoOpNoOp^Conv2D/ReadVariableOp<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ҟ
?
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9729
input_1%
conv2d_9606:	&
batch_normalization_9610:	&
batch_normalization_9612:	&
batch_normalization_9614:	&
batch_normalization_9616:	'
conv2d_1_9619:	(
batch_normalization_1_9623:(
batch_normalization_1_9625:(
batch_normalization_1_9627:(
batch_normalization_1_9629:'
conv2d_2_9632:(
batch_normalization_2_9636:(
batch_normalization_2_9638:(
batch_normalization_2_9640:(
batch_normalization_2_9642:'
conv2d_3_9645: (
batch_normalization_3_9649: (
batch_normalization_3_9651: (
batch_normalization_3_9653: (
batch_normalization_3_9655: '
conv2d_4_9658: 0(
batch_normalization_4_9662:0(
batch_normalization_4_9664:0(
batch_normalization_4_9666:0(
batch_normalization_4_9668:0'
conv2d_5_9672:0@(
batch_normalization_5_9676:@(
batch_normalization_5_9678:@(
batch_normalization_5_9680:@(
batch_normalization_5_9682:@

dense_9686:	?


dense_9688:

identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_9606*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_8717?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_8726?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_9610batch_normalization_9612batch_normalization_9614batch_normalization_9616*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8354?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_9619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8750?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_8759?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_1_9623batch_normalization_1_9625batch_normalization_1_9627batch_normalization_1_9629*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8418?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_9632*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8783?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_8792?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_2_9636batch_normalization_2_9638batch_normalization_2_9640batch_normalization_2_9642*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8482?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_9645*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8816?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_8825?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_3_9649batch_normalization_3_9651batch_normalization_3_9653batch_normalization_3_9655*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8546?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_9658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8849?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_8858?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_4_9662batch_normalization_4_9664batch_normalization_4_9666batch_normalization_4_9668*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8610?
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8630?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_5_9672*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_8883?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_8892?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_5_9676batch_normalization_5_9678batch_normalization_5_9680batch_normalization_5_9682*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8686?
flatten/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8909?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_9686
dense_9688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8921?
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_softmax_layer_call_and_return_conditional_losses_8932?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9606*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_9619*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_9632*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_9645*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_9658*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_9672*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8783

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_activation_4_layer_call_fn_10825

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_8858h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10526

inputs8
conv2d_readvariableop_resource:	
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_10402`
Fccnn_custom_conv2d_5_kernel_regularizer_square_readvariableop_resource:0@
identity??=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFccnn_custom_conv2d_5_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/ccnn_custom/conv2d_5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp
?
b
F__inference_activation_1_layer_call_and_return_conditional_losses_8759

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10678

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8579

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_10513

inputs!
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8750w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10820

inputs8
conv2d_readvariableop_resource: 0
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
C
'__inference_softmax_layer_call_fn_10331

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_softmax_layer_call_and_return_conditional_losses_8932`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_10347^
Dccnn_custom_conv2d_kernel_regularizer_square_readvariableop_resource:	
identity??;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDccnn_custom_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentity-ccnn_custom/conv2d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp
?
?
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8686

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_10451

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8323?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_10415

inputs!
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_8717w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8387

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8630

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_10301

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8909a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8849

inputs8
conv2d_readvariableop_resource: 0
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????0?
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_3_layer_call_fn_10745

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8515?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_5_layer_call_fn_10964

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8686?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8482

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_8717

inputs8
conv2d_readvariableop_resource:	
identity??Conv2D/ReadVariableOp?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingSAME*
strides
?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????	?
NoOpNoOp^Conv2D/ReadVariableOp<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_2_layer_call_fn_10660

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8482?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_8921

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10500

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
(__inference_conv2d_5_layer_call_fn_10915

inputs!
unknown:0@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_8883w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????0: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_1_layer_call_fn_10562

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8418?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?#
F__inference_ccnn_custom_layer_call_and_return_conditional_losses_10225
p_tinput?
%conv2d_conv2d_readvariableop_resource:	9
+batch_normalization_readvariableop_resource:	;
-batch_normalization_readvariableop_1_resource:	J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:	L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	A
'conv2d_1_conv2d_readvariableop_resource:	;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_2_conv2d_readvariableop_resource:;
-batch_normalization_2_readvariableop_resource:=
/batch_normalization_2_readvariableop_1_resource:L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_3_conv2d_readvariableop_resource: ;
-batch_normalization_3_readvariableop_resource: =
/batch_normalization_3_readvariableop_1_resource: L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_4_conv2d_readvariableop_resource: 0;
-batch_normalization_4_readvariableop_resource:0=
/batch_normalization_4_readvariableop_1_resource:0L
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_5_conv2d_readvariableop_resource:0@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@7
$dense_matmul_readvariableop_resource:	?
3
%dense_biasadd_readvariableop_resource:

identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1?$batch_normalization_3/AssignNewValue?&batch_normalization_3/AssignNewValue_1?5batch_normalization_3/FusedBatchNormV3/ReadVariableOp?7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_3/ReadVariableOp?&batch_normalization_3/ReadVariableOp_1?$batch_normalization_4/AssignNewValue?&batch_normalization_4/AssignNewValue_1?5batch_normalization_4/FusedBatchNormV3/ReadVariableOp?7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_4/ReadVariableOp?&batch_normalization_4/ReadVariableOp_1?$batch_normalization_5/AssignNewValue?&batch_normalization_5/AssignNewValue_1?5batch_normalization_5/FusedBatchNormV3/ReadVariableOp?7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_5/ReadVariableOp?&batch_normalization_5/ReadVariableOp_1?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
conv2d/Conv2DConv2Dp_tinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingSAME*
strides
i
activation/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:?????????	?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:	*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????	:	:	:	:	:*
epsilon%o?:*
exponential_avg_factor%
?#<?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
m
activation_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:??????????
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
m
activation_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:??????????
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
m
activation_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:????????? ?
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_4/Conv2DConv2D*batch_normalization_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
m
activation_4/ReluReluconv2d_4/Conv2D:output:0*
T0*/
_output_shapes
:?????????0?
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:0*
dtype0?
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
max_pooling2d/MaxPoolMaxPool*batch_normalization_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
conv2d_5/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
m
activation_5/ReluReluconv2d_5/Conv2D:output:0*
T0*/
_output_shapes
:?????????@?
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten/ReshapeReshape*batch_normalization_5/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: h
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
p_tInput
?
?
(__inference_conv2d_3_layer_call_fn_10709

inputs!
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8816w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
F__inference_activation_2_layer_call_and_return_conditional_losses_8792

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8515

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?"
__inference__wrapped_model_8301
input_1K
1ccnn_custom_conv2d_conv2d_readvariableop_resource:	E
7ccnn_custom_batch_normalization_readvariableop_resource:	G
9ccnn_custom_batch_normalization_readvariableop_1_resource:	V
Hccnn_custom_batch_normalization_fusedbatchnormv3_readvariableop_resource:	X
Jccnn_custom_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:	M
3ccnn_custom_conv2d_1_conv2d_readvariableop_resource:	G
9ccnn_custom_batch_normalization_1_readvariableop_resource:I
;ccnn_custom_batch_normalization_1_readvariableop_1_resource:X
Jccnn_custom_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:Z
Lccnn_custom_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:M
3ccnn_custom_conv2d_2_conv2d_readvariableop_resource:G
9ccnn_custom_batch_normalization_2_readvariableop_resource:I
;ccnn_custom_batch_normalization_2_readvariableop_1_resource:X
Jccnn_custom_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:Z
Lccnn_custom_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:M
3ccnn_custom_conv2d_3_conv2d_readvariableop_resource: G
9ccnn_custom_batch_normalization_3_readvariableop_resource: I
;ccnn_custom_batch_normalization_3_readvariableop_1_resource: X
Jccnn_custom_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: Z
Lccnn_custom_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: M
3ccnn_custom_conv2d_4_conv2d_readvariableop_resource: 0G
9ccnn_custom_batch_normalization_4_readvariableop_resource:0I
;ccnn_custom_batch_normalization_4_readvariableop_1_resource:0X
Jccnn_custom_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:0Z
Lccnn_custom_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:0M
3ccnn_custom_conv2d_5_conv2d_readvariableop_resource:0@G
9ccnn_custom_batch_normalization_5_readvariableop_resource:@I
;ccnn_custom_batch_normalization_5_readvariableop_1_resource:@X
Jccnn_custom_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@Z
Lccnn_custom_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@C
0ccnn_custom_dense_matmul_readvariableop_resource:	?
?
1ccnn_custom_dense_biasadd_readvariableop_resource:

identity???ccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp?Accnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?.ccnn_custom/batch_normalization/ReadVariableOp?0ccnn_custom/batch_normalization/ReadVariableOp_1?Accnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Cccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?0ccnn_custom/batch_normalization_1/ReadVariableOp?2ccnn_custom/batch_normalization_1/ReadVariableOp_1?Accnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp?Cccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?0ccnn_custom/batch_normalization_2/ReadVariableOp?2ccnn_custom/batch_normalization_2/ReadVariableOp_1?Accnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp?Cccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?0ccnn_custom/batch_normalization_3/ReadVariableOp?2ccnn_custom/batch_normalization_3/ReadVariableOp_1?Accnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp?Cccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?0ccnn_custom/batch_normalization_4/ReadVariableOp?2ccnn_custom/batch_normalization_4/ReadVariableOp_1?Accnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp?Cccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?0ccnn_custom/batch_normalization_5/ReadVariableOp?2ccnn_custom/batch_normalization_5/ReadVariableOp_1?(ccnn_custom/conv2d/Conv2D/ReadVariableOp?*ccnn_custom/conv2d_1/Conv2D/ReadVariableOp?*ccnn_custom/conv2d_2/Conv2D/ReadVariableOp?*ccnn_custom/conv2d_3/Conv2D/ReadVariableOp?*ccnn_custom/conv2d_4/Conv2D/ReadVariableOp?*ccnn_custom/conv2d_5/Conv2D/ReadVariableOp?(ccnn_custom/dense/BiasAdd/ReadVariableOp?'ccnn_custom/dense/MatMul/ReadVariableOp?
(ccnn_custom/conv2d/Conv2D/ReadVariableOpReadVariableOp1ccnn_custom_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
ccnn_custom/conv2d/Conv2DConv2Dinput_10ccnn_custom/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingSAME*
strides
?
ccnn_custom/activation/ReluRelu"ccnn_custom/conv2d/Conv2D:output:0*
T0*/
_output_shapes
:?????????	?
.ccnn_custom/batch_normalization/ReadVariableOpReadVariableOp7ccnn_custom_batch_normalization_readvariableop_resource*
_output_shapes
:	*
dtype0?
0ccnn_custom/batch_normalization/ReadVariableOp_1ReadVariableOp9ccnn_custom_batch_normalization_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
?ccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHccnn_custom_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
Accnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJccnn_custom_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
0ccnn_custom/batch_normalization/FusedBatchNormV3FusedBatchNormV3)ccnn_custom/activation/Relu:activations:06ccnn_custom/batch_normalization/ReadVariableOp:value:08ccnn_custom/batch_normalization/ReadVariableOp_1:value:0Gccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Iccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????	:	:	:	:	:*
epsilon%o?:*
is_training( ?
*ccnn_custom/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3ccnn_custom_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
ccnn_custom/conv2d_1/Conv2DConv2D4ccnn_custom/batch_normalization/FusedBatchNormV3:y:02ccnn_custom/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
ccnn_custom/activation_1/ReluRelu$ccnn_custom/conv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:??????????
0ccnn_custom/batch_normalization_1/ReadVariableOpReadVariableOp9ccnn_custom_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0?
2ccnn_custom/batch_normalization_1/ReadVariableOp_1ReadVariableOp;ccnn_custom_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Accnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJccnn_custom_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Cccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLccnn_custom_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
2ccnn_custom/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3+ccnn_custom/activation_1/Relu:activations:08ccnn_custom/batch_normalization_1/ReadVariableOp:value:0:ccnn_custom/batch_normalization_1/ReadVariableOp_1:value:0Iccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Kccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
*ccnn_custom/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3ccnn_custom_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
ccnn_custom/conv2d_2/Conv2DConv2D6ccnn_custom/batch_normalization_1/FusedBatchNormV3:y:02ccnn_custom/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
ccnn_custom/activation_2/ReluRelu$ccnn_custom/conv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:??????????
0ccnn_custom/batch_normalization_2/ReadVariableOpReadVariableOp9ccnn_custom_batch_normalization_2_readvariableop_resource*
_output_shapes
:*
dtype0?
2ccnn_custom/batch_normalization_2/ReadVariableOp_1ReadVariableOp;ccnn_custom_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Accnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJccnn_custom_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Cccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLccnn_custom_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
2ccnn_custom/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3+ccnn_custom/activation_2/Relu:activations:08ccnn_custom/batch_normalization_2/ReadVariableOp:value:0:ccnn_custom/batch_normalization_2/ReadVariableOp_1:value:0Iccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Kccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
*ccnn_custom/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3ccnn_custom_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
ccnn_custom/conv2d_3/Conv2DConv2D6ccnn_custom/batch_normalization_2/FusedBatchNormV3:y:02ccnn_custom/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
ccnn_custom/activation_3/ReluRelu$ccnn_custom/conv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:????????? ?
0ccnn_custom/batch_normalization_3/ReadVariableOpReadVariableOp9ccnn_custom_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0?
2ccnn_custom/batch_normalization_3/ReadVariableOp_1ReadVariableOp;ccnn_custom_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
Accnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJccnn_custom_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Cccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLccnn_custom_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
2ccnn_custom/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3+ccnn_custom/activation_3/Relu:activations:08ccnn_custom/batch_normalization_3/ReadVariableOp:value:0:ccnn_custom/batch_normalization_3/ReadVariableOp_1:value:0Iccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Kccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
*ccnn_custom/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3ccnn_custom_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
ccnn_custom/conv2d_4/Conv2DConv2D6ccnn_custom/batch_normalization_3/FusedBatchNormV3:y:02ccnn_custom/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
ccnn_custom/activation_4/ReluRelu$ccnn_custom/conv2d_4/Conv2D:output:0*
T0*/
_output_shapes
:?????????0?
0ccnn_custom/batch_normalization_4/ReadVariableOpReadVariableOp9ccnn_custom_batch_normalization_4_readvariableop_resource*
_output_shapes
:0*
dtype0?
2ccnn_custom/batch_normalization_4/ReadVariableOp_1ReadVariableOp;ccnn_custom_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
Accnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJccnn_custom_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
Cccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLccnn_custom_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
2ccnn_custom/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3+ccnn_custom/activation_4/Relu:activations:08ccnn_custom/batch_normalization_4/ReadVariableOp:value:0:ccnn_custom/batch_normalization_4/ReadVariableOp_1:value:0Iccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Kccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
is_training( ?
!ccnn_custom/max_pooling2d/MaxPoolMaxPool6ccnn_custom/batch_normalization_4/FusedBatchNormV3:y:0*/
_output_shapes
:?????????0*
ksize
*
paddingVALID*
strides
?
*ccnn_custom/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3ccnn_custom_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
ccnn_custom/conv2d_5/Conv2DConv2D*ccnn_custom/max_pooling2d/MaxPool:output:02ccnn_custom/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
ccnn_custom/activation_5/ReluRelu$ccnn_custom/conv2d_5/Conv2D:output:0*
T0*/
_output_shapes
:?????????@?
0ccnn_custom/batch_normalization_5/ReadVariableOpReadVariableOp9ccnn_custom_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0?
2ccnn_custom/batch_normalization_5/ReadVariableOp_1ReadVariableOp;ccnn_custom_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Accnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJccnn_custom_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Cccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLccnn_custom_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
2ccnn_custom/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+ccnn_custom/activation_5/Relu:activations:08ccnn_custom/batch_normalization_5/ReadVariableOp:value:0:ccnn_custom/batch_normalization_5/ReadVariableOp_1:value:0Iccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Kccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( j
ccnn_custom/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
ccnn_custom/flatten/ReshapeReshape6ccnn_custom/batch_normalization_5/FusedBatchNormV3:y:0"ccnn_custom/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
'ccnn_custom/dense/MatMul/ReadVariableOpReadVariableOp0ccnn_custom_dense_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
ccnn_custom/dense/MatMulMatMul$ccnn_custom/flatten/Reshape:output:0/ccnn_custom/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
(ccnn_custom/dense/BiasAdd/ReadVariableOpReadVariableOp1ccnn_custom_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
ccnn_custom/dense/BiasAddBiasAdd"ccnn_custom/dense/MatMul:product:00ccnn_custom/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
|
ccnn_custom/softmax/SoftmaxSoftmax"ccnn_custom/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
t
IdentityIdentity%ccnn_custom/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp@^ccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOpB^ccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/^ccnn_custom/batch_normalization/ReadVariableOp1^ccnn_custom/batch_normalization/ReadVariableOp_1B^ccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOpD^ccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_11^ccnn_custom/batch_normalization_1/ReadVariableOp3^ccnn_custom/batch_normalization_1/ReadVariableOp_1B^ccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^ccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^ccnn_custom/batch_normalization_2/ReadVariableOp3^ccnn_custom/batch_normalization_2/ReadVariableOp_1B^ccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^ccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^ccnn_custom/batch_normalization_3/ReadVariableOp3^ccnn_custom/batch_normalization_3/ReadVariableOp_1B^ccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^ccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^ccnn_custom/batch_normalization_4/ReadVariableOp3^ccnn_custom/batch_normalization_4/ReadVariableOp_1B^ccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^ccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^ccnn_custom/batch_normalization_5/ReadVariableOp3^ccnn_custom/batch_normalization_5/ReadVariableOp_1)^ccnn_custom/conv2d/Conv2D/ReadVariableOp+^ccnn_custom/conv2d_1/Conv2D/ReadVariableOp+^ccnn_custom/conv2d_2/Conv2D/ReadVariableOp+^ccnn_custom/conv2d_3/Conv2D/ReadVariableOp+^ccnn_custom/conv2d_4/Conv2D/ReadVariableOp+^ccnn_custom/conv2d_5/Conv2D/ReadVariableOp)^ccnn_custom/dense/BiasAdd/ReadVariableOp(^ccnn_custom/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
?ccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp?ccnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Accnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Accnn_custom/batch_normalization/FusedBatchNormV3/ReadVariableOp_12`
.ccnn_custom/batch_normalization/ReadVariableOp.ccnn_custom/batch_normalization/ReadVariableOp2d
0ccnn_custom/batch_normalization/ReadVariableOp_10ccnn_custom/batch_normalization/ReadVariableOp_12?
Accnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOpAccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Cccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Cccnn_custom/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12d
0ccnn_custom/batch_normalization_1/ReadVariableOp0ccnn_custom/batch_normalization_1/ReadVariableOp2h
2ccnn_custom/batch_normalization_1/ReadVariableOp_12ccnn_custom/batch_normalization_1/ReadVariableOp_12?
Accnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
Cccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Cccnn_custom/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0ccnn_custom/batch_normalization_2/ReadVariableOp0ccnn_custom/batch_normalization_2/ReadVariableOp2h
2ccnn_custom/batch_normalization_2/ReadVariableOp_12ccnn_custom/batch_normalization_2/ReadVariableOp_12?
Accnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2?
Cccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Cccnn_custom/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0ccnn_custom/batch_normalization_3/ReadVariableOp0ccnn_custom/batch_normalization_3/ReadVariableOp2h
2ccnn_custom/batch_normalization_3/ReadVariableOp_12ccnn_custom/batch_normalization_3/ReadVariableOp_12?
Accnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2?
Cccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Cccnn_custom/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0ccnn_custom/batch_normalization_4/ReadVariableOp0ccnn_custom/batch_normalization_4/ReadVariableOp2h
2ccnn_custom/batch_normalization_4/ReadVariableOp_12ccnn_custom/batch_normalization_4/ReadVariableOp_12?
Accnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2?
Cccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Cccnn_custom/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0ccnn_custom/batch_normalization_5/ReadVariableOp0ccnn_custom/batch_normalization_5/ReadVariableOp2h
2ccnn_custom/batch_normalization_5/ReadVariableOp_12ccnn_custom/batch_normalization_5/ReadVariableOp_12T
(ccnn_custom/conv2d/Conv2D/ReadVariableOp(ccnn_custom/conv2d/Conv2D/ReadVariableOp2X
*ccnn_custom/conv2d_1/Conv2D/ReadVariableOp*ccnn_custom/conv2d_1/Conv2D/ReadVariableOp2X
*ccnn_custom/conv2d_2/Conv2D/ReadVariableOp*ccnn_custom/conv2d_2/Conv2D/ReadVariableOp2X
*ccnn_custom/conv2d_3/Conv2D/ReadVariableOp*ccnn_custom/conv2d_3/Conv2D/ReadVariableOp2X
*ccnn_custom/conv2d_4/Conv2D/ReadVariableOp*ccnn_custom/conv2d_4/Conv2D/ReadVariableOp2X
*ccnn_custom/conv2d_5/Conv2D/ReadVariableOp*ccnn_custom/conv2d_5/Conv2D/ReadVariableOp2T
(ccnn_custom/dense/BiasAdd/ReadVariableOp(ccnn_custom/dense/BiasAdd/ReadVariableOp2R
'ccnn_custom/dense/MatMul/ReadVariableOp'ccnn_custom/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8546

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_10732

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8750

inputs8
conv2d_readvariableop_resource:	
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????	: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
b
F__inference_activation_4_layer_call_and_return_conditional_losses_8858

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
H
,__inference_activation_2_layer_call_fn_10629

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_8792h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_activation_1_layer_call_fn_10531

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_8759h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10902

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_10326

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_1_layer_call_fn_10549

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8387?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_10307

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_4_layer_call_fn_10807

inputs!
unknown: 0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8849w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:????????? : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_10391`
Fccnn_custom_conv2d_4_kernel_regularizer_square_readvariableop_resource: 0
identity??=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFccnn_custom_conv2d_4_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentity/ccnn_custom/conv2d_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp
?
H
,__inference_activation_5_layer_call_fn_10933

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_8892h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_10634

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_4_layer_call_fn_10856

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8610?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
b
F__inference_activation_3_layer_call_and_return_conditional_losses_8825

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10776

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_11000

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
*__inference_ccnn_custom_layer_call_fn_9477
input_1!
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	#
	unknown_4:	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:$

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 0

unknown_20:0

unknown_21:0

unknown_22:0

unknown_23:0$

unknown_24:0@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	?


unknown_30:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_8909

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_10464

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8354?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10794

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
`
D__inference_activation_layer_call_and_return_conditional_losses_8726

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????	:W S
/
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8354

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10580

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8323

inputs%
readvariableop_resource:	'
readvariableop_1_resource:	6
(fusedbatchnormv3_readvariableop_resource:	8
*fusedbatchnormv3_readvariableop_1_resource:	
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:	*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:	*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????	:	:	:	:	:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????	?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????	: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????	
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8610

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
??
?=
!__inference__traced_restore_11519
file_prefix<
)assignvariableop_ccnn_custom_dense_kernel:	?
7
)assignvariableop_1_ccnn_custom_dense_bias:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: F
,assignvariableop_7_ccnn_custom_conv2d_kernel:	F
8assignvariableop_8_ccnn_custom_batch_normalization_gamma:	E
7assignvariableop_9_ccnn_custom_batch_normalization_beta:	I
/assignvariableop_10_ccnn_custom_conv2d_1_kernel:	I
;assignvariableop_11_ccnn_custom_batch_normalization_1_gamma:H
:assignvariableop_12_ccnn_custom_batch_normalization_1_beta:I
/assignvariableop_13_ccnn_custom_conv2d_2_kernel:I
;assignvariableop_14_ccnn_custom_batch_normalization_2_gamma:H
:assignvariableop_15_ccnn_custom_batch_normalization_2_beta:I
/assignvariableop_16_ccnn_custom_conv2d_3_kernel: I
;assignvariableop_17_ccnn_custom_batch_normalization_3_gamma: H
:assignvariableop_18_ccnn_custom_batch_normalization_3_beta: I
/assignvariableop_19_ccnn_custom_conv2d_4_kernel: 0I
;assignvariableop_20_ccnn_custom_batch_normalization_4_gamma:0H
:assignvariableop_21_ccnn_custom_batch_normalization_4_beta:0I
/assignvariableop_22_ccnn_custom_conv2d_5_kernel:0@I
;assignvariableop_23_ccnn_custom_batch_normalization_5_gamma:@H
:assignvariableop_24_ccnn_custom_batch_normalization_5_beta:@M
?assignvariableop_25_ccnn_custom_batch_normalization_moving_mean:	Q
Cassignvariableop_26_ccnn_custom_batch_normalization_moving_variance:	O
Aassignvariableop_27_ccnn_custom_batch_normalization_1_moving_mean:S
Eassignvariableop_28_ccnn_custom_batch_normalization_1_moving_variance:O
Aassignvariableop_29_ccnn_custom_batch_normalization_2_moving_mean:S
Eassignvariableop_30_ccnn_custom_batch_normalization_2_moving_variance:O
Aassignvariableop_31_ccnn_custom_batch_normalization_3_moving_mean: S
Eassignvariableop_32_ccnn_custom_batch_normalization_3_moving_variance: O
Aassignvariableop_33_ccnn_custom_batch_normalization_4_moving_mean:0S
Eassignvariableop_34_ccnn_custom_batch_normalization_4_moving_variance:0O
Aassignvariableop_35_ccnn_custom_batch_normalization_5_moving_mean:@S
Eassignvariableop_36_ccnn_custom_batch_normalization_5_moving_variance:@#
assignvariableop_37_total: #
assignvariableop_38_count: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: F
3assignvariableop_41_adam_ccnn_custom_dense_kernel_m:	?
?
1assignvariableop_42_adam_ccnn_custom_dense_bias_m:
N
4assignvariableop_43_adam_ccnn_custom_conv2d_kernel_m:	N
@assignvariableop_44_adam_ccnn_custom_batch_normalization_gamma_m:	M
?assignvariableop_45_adam_ccnn_custom_batch_normalization_beta_m:	P
6assignvariableop_46_adam_ccnn_custom_conv2d_1_kernel_m:	P
Bassignvariableop_47_adam_ccnn_custom_batch_normalization_1_gamma_m:O
Aassignvariableop_48_adam_ccnn_custom_batch_normalization_1_beta_m:P
6assignvariableop_49_adam_ccnn_custom_conv2d_2_kernel_m:P
Bassignvariableop_50_adam_ccnn_custom_batch_normalization_2_gamma_m:O
Aassignvariableop_51_adam_ccnn_custom_batch_normalization_2_beta_m:P
6assignvariableop_52_adam_ccnn_custom_conv2d_3_kernel_m: P
Bassignvariableop_53_adam_ccnn_custom_batch_normalization_3_gamma_m: O
Aassignvariableop_54_adam_ccnn_custom_batch_normalization_3_beta_m: P
6assignvariableop_55_adam_ccnn_custom_conv2d_4_kernel_m: 0P
Bassignvariableop_56_adam_ccnn_custom_batch_normalization_4_gamma_m:0O
Aassignvariableop_57_adam_ccnn_custom_batch_normalization_4_beta_m:0P
6assignvariableop_58_adam_ccnn_custom_conv2d_5_kernel_m:0@P
Bassignvariableop_59_adam_ccnn_custom_batch_normalization_5_gamma_m:@O
Aassignvariableop_60_adam_ccnn_custom_batch_normalization_5_beta_m:@F
3assignvariableop_61_adam_ccnn_custom_dense_kernel_v:	?
?
1assignvariableop_62_adam_ccnn_custom_dense_bias_v:
N
4assignvariableop_63_adam_ccnn_custom_conv2d_kernel_v:	N
@assignvariableop_64_adam_ccnn_custom_batch_normalization_gamma_v:	M
?assignvariableop_65_adam_ccnn_custom_batch_normalization_beta_v:	P
6assignvariableop_66_adam_ccnn_custom_conv2d_1_kernel_v:	P
Bassignvariableop_67_adam_ccnn_custom_batch_normalization_1_gamma_v:O
Aassignvariableop_68_adam_ccnn_custom_batch_normalization_1_beta_v:P
6assignvariableop_69_adam_ccnn_custom_conv2d_2_kernel_v:P
Bassignvariableop_70_adam_ccnn_custom_batch_normalization_2_gamma_v:O
Aassignvariableop_71_adam_ccnn_custom_batch_normalization_2_beta_v:P
6assignvariableop_72_adam_ccnn_custom_conv2d_3_kernel_v: P
Bassignvariableop_73_adam_ccnn_custom_batch_normalization_3_gamma_v: O
Aassignvariableop_74_adam_ccnn_custom_batch_normalization_3_beta_v: P
6assignvariableop_75_adam_ccnn_custom_conv2d_4_kernel_v: 0P
Bassignvariableop_76_adam_ccnn_custom_batch_normalization_4_gamma_v:0O
Aassignvariableop_77_adam_ccnn_custom_batch_normalization_4_beta_v:0P
6assignvariableop_78_adam_ccnn_custom_conv2d_5_kernel_v:0@P
Bassignvariableop_79_adam_ccnn_custom_batch_normalization_5_gamma_v:@O
Aassignvariableop_80_adam_ccnn_custom_batch_normalization_5_beta_v:@
identity_82??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_9?$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?#
value?#B?#RB-OutputLayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+OutputLayer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIOutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGOutputLayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIOutputLayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGOutputLayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:R*
dtype0*?
value?B?RB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*`
dtypesV
T2R	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp)assignvariableop_ccnn_custom_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_ccnn_custom_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp,assignvariableop_7_ccnn_custom_conv2d_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_ccnn_custom_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_ccnn_custom_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_ccnn_custom_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp;assignvariableop_11_ccnn_custom_batch_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp:assignvariableop_12_ccnn_custom_batch_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_ccnn_custom_conv2d_2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp;assignvariableop_14_ccnn_custom_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_ccnn_custom_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_ccnn_custom_conv2d_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp;assignvariableop_17_ccnn_custom_batch_normalization_3_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp:assignvariableop_18_ccnn_custom_batch_normalization_3_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_ccnn_custom_conv2d_4_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp;assignvariableop_20_ccnn_custom_batch_normalization_4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp:assignvariableop_21_ccnn_custom_batch_normalization_4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_ccnn_custom_conv2d_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp;assignvariableop_23_ccnn_custom_batch_normalization_5_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp:assignvariableop_24_ccnn_custom_batch_normalization_5_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp?assignvariableop_25_ccnn_custom_batch_normalization_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpCassignvariableop_26_ccnn_custom_batch_normalization_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpAassignvariableop_27_ccnn_custom_batch_normalization_1_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpEassignvariableop_28_ccnn_custom_batch_normalization_1_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpAassignvariableop_29_ccnn_custom_batch_normalization_2_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpEassignvariableop_30_ccnn_custom_batch_normalization_2_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpAassignvariableop_31_ccnn_custom_batch_normalization_3_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpEassignvariableop_32_ccnn_custom_batch_normalization_3_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpAassignvariableop_33_ccnn_custom_batch_normalization_4_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpEassignvariableop_34_ccnn_custom_batch_normalization_4_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpAassignvariableop_35_ccnn_custom_batch_normalization_5_moving_meanIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpEassignvariableop_36_ccnn_custom_batch_normalization_5_moving_varianceIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_ccnn_custom_dense_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_ccnn_custom_dense_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_ccnn_custom_conv2d_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp@assignvariableop_44_adam_ccnn_custom_batch_normalization_gamma_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_ccnn_custom_batch_normalization_beta_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_ccnn_custom_conv2d_1_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpBassignvariableop_47_adam_ccnn_custom_batch_normalization_1_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpAassignvariableop_48_adam_ccnn_custom_batch_normalization_1_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_ccnn_custom_conv2d_2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpBassignvariableop_50_adam_ccnn_custom_batch_normalization_2_gamma_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpAassignvariableop_51_adam_ccnn_custom_batch_normalization_2_beta_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_ccnn_custom_conv2d_3_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpBassignvariableop_53_adam_ccnn_custom_batch_normalization_3_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpAassignvariableop_54_adam_ccnn_custom_batch_normalization_3_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_ccnn_custom_conv2d_4_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpBassignvariableop_56_adam_ccnn_custom_batch_normalization_4_gamma_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpAassignvariableop_57_adam_ccnn_custom_batch_normalization_4_beta_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_ccnn_custom_conv2d_5_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpBassignvariableop_59_adam_ccnn_custom_batch_normalization_5_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpAassignvariableop_60_adam_ccnn_custom_batch_normalization_5_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp3assignvariableop_61_adam_ccnn_custom_dense_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp1assignvariableop_62_adam_ccnn_custom_dense_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adam_ccnn_custom_conv2d_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp@assignvariableop_64_adam_ccnn_custom_batch_normalization_gamma_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_ccnn_custom_batch_normalization_beta_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_ccnn_custom_conv2d_1_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpBassignvariableop_67_adam_ccnn_custom_batch_normalization_1_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOpAassignvariableop_68_adam_ccnn_custom_batch_normalization_1_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_ccnn_custom_conv2d_2_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpBassignvariableop_70_adam_ccnn_custom_batch_normalization_2_gamma_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpAassignvariableop_71_adam_ccnn_custom_batch_normalization_2_beta_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_ccnn_custom_conv2d_3_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpBassignvariableop_73_adam_ccnn_custom_batch_normalization_3_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpAassignvariableop_74_adam_ccnn_custom_batch_normalization_3_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_ccnn_custom_conv2d_4_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOpBassignvariableop_76_adam_ccnn_custom_batch_normalization_4_gamma_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOpAassignvariableop_77_adam_ccnn_custom_batch_normalization_4_beta_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_ccnn_custom_conv2d_5_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOpBassignvariableop_79_adam_ccnn_custom_batch_normalization_5_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOpAassignvariableop_80_adam_ccnn_custom_batch_normalization_5_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_81Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_82IdentityIdentity_81:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_82Identity_82:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_8971
p_tinput%
conv2d_8718:	&
batch_normalization_8728:	&
batch_normalization_8730:	&
batch_normalization_8732:	&
batch_normalization_8734:	'
conv2d_1_8751:	(
batch_normalization_1_8761:(
batch_normalization_1_8763:(
batch_normalization_1_8765:(
batch_normalization_1_8767:'
conv2d_2_8784:(
batch_normalization_2_8794:(
batch_normalization_2_8796:(
batch_normalization_2_8798:(
batch_normalization_2_8800:'
conv2d_3_8817: (
batch_normalization_3_8827: (
batch_normalization_3_8829: (
batch_normalization_3_8831: (
batch_normalization_3_8833: '
conv2d_4_8850: 0(
batch_normalization_4_8860:0(
batch_normalization_4_8862:0(
batch_normalization_4_8864:0(
batch_normalization_4_8866:0'
conv2d_5_8884:0@(
batch_normalization_5_8894:@(
batch_normalization_5_8896:@(
batch_normalization_5_8898:@(
batch_normalization_5_8900:@

dense_8922:	?


dense_8924:

identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?-batch_normalization_3/StatefulPartitionedCall?-batch_normalization_4/StatefulPartitionedCall?-batch_normalization_5/StatefulPartitionedCall?;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallp_tinputconv2d_8718*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_8717?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_8726?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0batch_normalization_8728batch_normalization_8730batch_normalization_8732batch_normalization_8734*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_8323?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_8751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8750?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_8759?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_1_8761batch_normalization_1_8763batch_normalization_1_8765batch_normalization_1_8767*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8387?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_8784*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8783?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_8792?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_2_8794batch_normalization_2_8796batch_normalization_2_8798batch_normalization_2_8800*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8451?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_8817*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8816?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_8825?
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_3_8827batch_normalization_3_8829batch_normalization_3_8831batch_normalization_3_8833*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_8515?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0conv2d_4_8850*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8849?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_8858?
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_4_8860batch_normalization_4_8862batch_normalization_4_8864batch_normalization_4_8866*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_8579?
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8630?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_5_8884*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_8883?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_8892?
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_5_8894batch_normalization_5_8896batch_normalization_5_8898batch_normalization_5_8900*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_8655?
flatten/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_8909?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_8922
dense_8924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_8921?
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_softmax_layer_call_and_return_conditional_losses_8932?
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8718*&
_output_shapes
:	*
dtype0?
,ccnn_custom/conv2d/kernel/Regularizer/SquareSquareCccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
+ccnn_custom/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
)ccnn_custom/conv2d/kernel/Regularizer/SumSum0ccnn_custom/conv2d/kernel/Regularizer/Square:y:04ccnn_custom/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: p
+ccnn_custom/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
)ccnn_custom/conv2d/kernel/Regularizer/mulMul4ccnn_custom/conv2d/kernel/Regularizer/mul/x:output:02ccnn_custom/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_8751*&
_output_shapes
:	*
dtype0?
.ccnn_custom/conv2d_1/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:	?
-ccnn_custom/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_1/kernel/Regularizer/SumSum2ccnn_custom/conv2d_1/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_1/kernel/Regularizer/mulMul6ccnn_custom/conv2d_1/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_8784*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_8817*&
_output_shapes
: *
dtype0?
.ccnn_custom/conv2d_3/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-ccnn_custom/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_3/kernel/Regularizer/SumSum2ccnn_custom/conv2d_3/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_3/kernel/Regularizer/mulMul6ccnn_custom/conv2d_3/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_8850*&
_output_shapes
: 0*
dtype0?
.ccnn_custom/conv2d_4/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-ccnn_custom/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_4/kernel/Regularizer/SumSum2ccnn_custom/conv2d_4/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_4/kernel/Regularizer/mulMul6ccnn_custom/conv2d_4/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_8884*&
_output_shapes
:0@*
dtype0?
.ccnn_custom/conv2d_5/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0@?
-ccnn_custom/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_5/kernel/Regularizer/SumSum2ccnn_custom/conv2d_5/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_5/kernel/Regularizer/mulMul6ccnn_custom/conv2d_5/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall<^ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp>^ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2z
;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp;ccnn_custom/conv2d/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2~
=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_5/kernel/Regularizer/Square/ReadVariableOp2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
p_tInput
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_10830

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10696

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10624

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp?=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
.ccnn_custom/conv2d_2/kernel/Regularizer/SquareSquareEccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-ccnn_custom/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             ?
+ccnn_custom/conv2d_2/kernel/Regularizer/SumSum2ccnn_custom/conv2d_2/kernel/Regularizer/Square:y:06ccnn_custom/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: r
-ccnn_custom/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
+ccnn_custom/conv2d_2/kernel/Regularizer/mulMul6ccnn_custom/conv2d_2/kernel/Regularizer/mul/x:output:04ccnn_custom/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^Conv2D/ReadVariableOp>^ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp=ccnn_custom/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_10296
input_1!
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	#
	unknown_4:	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:$

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 0

unknown_20:0

unknown_21:0

unknown_22:0

unknown_23:0$

unknown_24:0@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	?


unknown_30:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_8301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_ccnn_custom_layer_call_fn_9840
p_tinput!
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	#
	unknown_4:	
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:$

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 0

unknown_20:0

unknown_21:0

unknown_22:0

unknown_23:0$

unknown_24:0@

unknown_25:@

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:	?


unknown_30:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallp_tinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_8971o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
p_tInput
?
?
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_8451

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?

Config

InputShape
ConvLayerFeatures
ConvWindows
PoolWindows
KerasLayers
FlatteningLayer
OutputLayer
	SoftmaxActivation

	optimizer
	Structure
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
z
CNN.InputShape
CNN.ConvOutputFeatures
CNN.ConvWindows
CNN.PoolWindows"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
4"
trackable_list_wrapper
?
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18"
trackable_list_wrapper
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
?

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Biter

Cbeta_1

Dbeta_2
	Edecay
Flearning_rate4m?5m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?4v?5v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?"
	optimizer
?
G0
H1
I2
J3
K4
L5
M6
N7
O8
P9
Q10
R11
S12
T13
U14
V15
W16
X17
Y18
Z19
[20
\21"
trackable_list_wrapper
?
]0
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12
j13
k14
l15
m16
n17
o18
p19
q20
r21
s22
t23
u24
v25
w26
x27
y28
z29
430
531"
trackable_list_wrapper
?
]0
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12
j13
k14
l15
m16
n17
418
519"
trackable_list_wrapper
K
{0
|1
}2
~3
4
?5"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_ccnn_custom_layer_call_fn_9038
*__inference_ccnn_custom_layer_call_fn_9840
*__inference_ccnn_custom_layer_call_fn_9909
*__inference_ccnn_custom_layer_call_fn_9477?
???
FullArgSpec+
args#? 
jself

jp_tInput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_ccnn_custom_layer_call_and_return_conditional_losses_10067
F__inference_ccnn_custom_layer_call_and_return_conditional_losses_10225
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9603
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9729?
???
FullArgSpec+
args#? 
jself

jp_tInput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference__wrapped_model_8301input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

]kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	^gamma
_beta
omoving_mean
pmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

`kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	agamma
bbeta
qmoving_mean
rmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ckernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	dgamma
ebeta
smoving_mean
tmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

fkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	ggamma
hbeta
umoving_mean
vmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ikernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	jgamma
kbeta
wmoving_mean
xmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

lkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	mgamma
nbeta
ymoving_mean
zmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_flatten_layer_call_fn_10301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_10307?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
+:)	?
2ccnn_custom/dense/kernel
$:"
2ccnn_custom/dense/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_dense_layer_call_fn_10316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_10326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_softmax_layer_call_fn_10331?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_softmax_layer_call_and_return_conditional_losses_10336?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
3:1	2ccnn_custom/conv2d/kernel
3:1	2%ccnn_custom/batch_normalization/gamma
2:0	2$ccnn_custom/batch_normalization/beta
5:3	2ccnn_custom/conv2d_1/kernel
5:32'ccnn_custom/batch_normalization_1/gamma
4:22&ccnn_custom/batch_normalization_1/beta
5:32ccnn_custom/conv2d_2/kernel
5:32'ccnn_custom/batch_normalization_2/gamma
4:22&ccnn_custom/batch_normalization_2/beta
5:3 2ccnn_custom/conv2d_3/kernel
5:3 2'ccnn_custom/batch_normalization_3/gamma
4:2 2&ccnn_custom/batch_normalization_3/beta
5:3 02ccnn_custom/conv2d_4/kernel
5:302'ccnn_custom/batch_normalization_4/gamma
4:202&ccnn_custom/batch_normalization_4/beta
5:30@2ccnn_custom/conv2d_5/kernel
5:3@2'ccnn_custom/batch_normalization_5/gamma
4:2@2&ccnn_custom/batch_normalization_5/beta
;:9	 (2+ccnn_custom/batch_normalization/moving_mean
?:=	 (2/ccnn_custom/batch_normalization/moving_variance
=:; (2-ccnn_custom/batch_normalization_1/moving_mean
A:? (21ccnn_custom/batch_normalization_1/moving_variance
=:; (2-ccnn_custom/batch_normalization_2/moving_mean
A:? (21ccnn_custom/batch_normalization_2/moving_variance
=:;  (2-ccnn_custom/batch_normalization_3/moving_mean
A:?  (21ccnn_custom/batch_normalization_3/moving_variance
=:;0 (2-ccnn_custom/batch_normalization_4/moving_mean
A:?0 (21ccnn_custom/batch_normalization_4/moving_variance
=:;@ (2-ccnn_custom/batch_normalization_5/moving_mean
A:?@ (21ccnn_custom/batch_normalization_5/moving_variance
?2?
__inference_loss_fn_0_10347?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_10358?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_10369?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_10380?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_10391?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_10402?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
v
o0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11"
trackable_list_wrapper
?
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18
19
20
	21"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_10296input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
]0"
trackable_list_wrapper
'
]0"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_conv2d_layer_call_fn_10415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_10428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_activation_layer_call_fn_10433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_activation_layer_call_and_return_conditional_losses_10438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
^0
_1
o2
p3"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_batch_normalization_layer_call_fn_10451
3__inference_batch_normalization_layer_call_fn_10464?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10482
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10500?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_1_layer_call_fn_10513?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_activation_1_layer_call_fn_10531?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_1_layer_call_and_return_conditional_losses_10536?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
a0
b1
q2
r3"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_1_layer_call_fn_10549
5__inference_batch_normalization_1_layer_call_fn_10562?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10580
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10598?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
'
c0"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_2_layer_call_fn_10611?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_activation_2_layer_call_fn_10629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_2_layer_call_and_return_conditional_losses_10634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
d0
e1
s2
t3"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_2_layer_call_fn_10647
5__inference_batch_normalization_2_layer_call_fn_10660?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10678
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10696?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
'
f0"
trackable_list_wrapper
'
f0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_3_layer_call_fn_10709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_activation_3_layer_call_fn_10727?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_3_layer_call_and_return_conditional_losses_10732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
g0
h1
u2
v3"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_3_layer_call_fn_10745
5__inference_batch_normalization_3_layer_call_fn_10758?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10776
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10794?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
'
i0"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_4_layer_call_fn_10807?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_activation_4_layer_call_fn_10825?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_4_layer_call_and_return_conditional_losses_10830?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
j0
k1
w2
x3"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_4_layer_call_fn_10843
5__inference_batch_normalization_4_layer_call_fn_10856?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10874
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10892?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_max_pooling2d_layer_call_fn_10897?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
l0"
trackable_list_wrapper
'
l0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_conv2d_5_layer_call_fn_10915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10928?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_activation_5_layer_call_fn_10933?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_5_layer_call_and_return_conditional_losses_10938?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
<
m0
n1
y2
z3"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_5_layer_call_fn_10951
5__inference_batch_normalization_5_layer_call_fn_10964?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10982
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_11000?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
|0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
0:.	?
2Adam/ccnn_custom/dense/kernel/m
):'
2Adam/ccnn_custom/dense/bias/m
8:6	2 Adam/ccnn_custom/conv2d/kernel/m
8:6	2,Adam/ccnn_custom/batch_normalization/gamma/m
7:5	2+Adam/ccnn_custom/batch_normalization/beta/m
::8	2"Adam/ccnn_custom/conv2d_1/kernel/m
::82.Adam/ccnn_custom/batch_normalization_1/gamma/m
9:72-Adam/ccnn_custom/batch_normalization_1/beta/m
::82"Adam/ccnn_custom/conv2d_2/kernel/m
::82.Adam/ccnn_custom/batch_normalization_2/gamma/m
9:72-Adam/ccnn_custom/batch_normalization_2/beta/m
::8 2"Adam/ccnn_custom/conv2d_3/kernel/m
::8 2.Adam/ccnn_custom/batch_normalization_3/gamma/m
9:7 2-Adam/ccnn_custom/batch_normalization_3/beta/m
::8 02"Adam/ccnn_custom/conv2d_4/kernel/m
::802.Adam/ccnn_custom/batch_normalization_4/gamma/m
9:702-Adam/ccnn_custom/batch_normalization_4/beta/m
::80@2"Adam/ccnn_custom/conv2d_5/kernel/m
::8@2.Adam/ccnn_custom/batch_normalization_5/gamma/m
9:7@2-Adam/ccnn_custom/batch_normalization_5/beta/m
0:.	?
2Adam/ccnn_custom/dense/kernel/v
):'
2Adam/ccnn_custom/dense/bias/v
8:6	2 Adam/ccnn_custom/conv2d/kernel/v
8:6	2,Adam/ccnn_custom/batch_normalization/gamma/v
7:5	2+Adam/ccnn_custom/batch_normalization/beta/v
::8	2"Adam/ccnn_custom/conv2d_1/kernel/v
::82.Adam/ccnn_custom/batch_normalization_1/gamma/v
9:72-Adam/ccnn_custom/batch_normalization_1/beta/v
::82"Adam/ccnn_custom/conv2d_2/kernel/v
::82.Adam/ccnn_custom/batch_normalization_2/gamma/v
9:72-Adam/ccnn_custom/batch_normalization_2/beta/v
::8 2"Adam/ccnn_custom/conv2d_3/kernel/v
::8 2.Adam/ccnn_custom/batch_normalization_3/gamma/v
9:7 2-Adam/ccnn_custom/batch_normalization_3/beta/v
::8 02"Adam/ccnn_custom/conv2d_4/kernel/v
::802.Adam/ccnn_custom/batch_normalization_4/gamma/v
9:702-Adam/ccnn_custom/batch_normalization_4/beta/v
::80@2"Adam/ccnn_custom/conv2d_5/kernel/v
::8@2.Adam/ccnn_custom/batch_normalization_5/gamma/v
9:7@2-Adam/ccnn_custom/batch_normalization_5/beta/v?
__inference__wrapped_model_8301? ]^_op`abqrcdestfghuvijkwxlmnyz458?5
.?+
)?&
input_1?????????
? "3?0
.
output_1"?
output_1?????????
?
G__inference_activation_1_layer_call_and_return_conditional_losses_10536h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_activation_1_layer_call_fn_10531[7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_activation_2_layer_call_and_return_conditional_losses_10634h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_activation_2_layer_call_fn_10629[7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_activation_3_layer_call_and_return_conditional_losses_10732h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
,__inference_activation_3_layer_call_fn_10727[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
G__inference_activation_4_layer_call_and_return_conditional_losses_10830h7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????0
? ?
,__inference_activation_4_layer_call_fn_10825[7?4
-?*
(?%
inputs?????????0
? " ??????????0?
G__inference_activation_5_layer_call_and_return_conditional_losses_10938h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
,__inference_activation_5_layer_call_fn_10933[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
E__inference_activation_layer_call_and_return_conditional_losses_10438h7?4
-?*
(?%
inputs?????????	
? "-?*
#? 
0?????????	
? ?
*__inference_activation_layer_call_fn_10433[7?4
-?*
(?%
inputs?????????	
? " ??????????	?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10580?abqrM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_10598?abqrM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
5__inference_batch_normalization_1_layer_call_fn_10549?abqrM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
5__inference_batch_normalization_1_layer_call_fn_10562?abqrM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10678?destM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10696?destM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
5__inference_batch_normalization_2_layer_call_fn_10647?destM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
5__inference_batch_normalization_2_layer_call_fn_10660?destM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10776?ghuvM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_10794?ghuvM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_batch_normalization_3_layer_call_fn_10745?ghuvM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_3_layer_call_fn_10758?ghuvM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10874?jkwxM?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_10892?jkwxM?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
5__inference_batch_normalization_4_layer_call_fn_10843?jkwxM?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
5__inference_batch_normalization_4_layer_call_fn_10856?jkwxM?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_10982?mnyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_11000?mnyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
5__inference_batch_normalization_5_layer_call_fn_10951?mnyzM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
5__inference_batch_normalization_5_layer_call_fn_10964?mnyzM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10482?^_opM?J
C?@
:?7
inputs+???????????????????????????	
p 
? "??<
5?2
0+???????????????????????????	
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10500?^_opM?J
C?@
:?7
inputs+???????????????????????????	
p
? "??<
5?2
0+???????????????????????????	
? ?
3__inference_batch_normalization_layer_call_fn_10451?^_opM?J
C?@
:?7
inputs+???????????????????????????	
p 
? "2?/+???????????????????????????	?
3__inference_batch_normalization_layer_call_fn_10464?^_opM?J
C?@
:?7
inputs+???????????????????????????	
p
? "2?/+???????????????????????????	?
F__inference_ccnn_custom_layer_call_and_return_conditional_losses_10067? ]^_op`abqrcdestfghuvijkwxlmnyz45=?:
3?0
*?'
p_tInput?????????
p 
? "%?"
?
0?????????

? ?
F__inference_ccnn_custom_layer_call_and_return_conditional_losses_10225? ]^_op`abqrcdestfghuvijkwxlmnyz45=?:
3?0
*?'
p_tInput?????????
p
? "%?"
?
0?????????

? ?
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9603? ]^_op`abqrcdestfghuvijkwxlmnyz45<?9
2?/
)?&
input_1?????????
p 
? "%?"
?
0?????????

? ?
E__inference_ccnn_custom_layer_call_and_return_conditional_losses_9729? ]^_op`abqrcdestfghuvijkwxlmnyz45<?9
2?/
)?&
input_1?????????
p
? "%?"
?
0?????????

? ?
*__inference_ccnn_custom_layer_call_fn_9038z ]^_op`abqrcdestfghuvijkwxlmnyz45<?9
2?/
)?&
input_1?????????
p 
? "??????????
?
*__inference_ccnn_custom_layer_call_fn_9477z ]^_op`abqrcdestfghuvijkwxlmnyz45<?9
2?/
)?&
input_1?????????
p
? "??????????
?
*__inference_ccnn_custom_layer_call_fn_9840{ ]^_op`abqrcdestfghuvijkwxlmnyz45=?:
3?0
*?'
p_tInput?????????
p 
? "??????????
?
*__inference_ccnn_custom_layer_call_fn_9909{ ]^_op`abqrcdestfghuvijkwxlmnyz45=?:
3?0
*?'
p_tInput?????????
p
? "??????????
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_10526k`7?4
-?*
(?%
inputs?????????	
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_1_layer_call_fn_10513^`7?4
-?*
(?%
inputs?????????	
? " ???????????
C__inference_conv2d_2_layer_call_and_return_conditional_losses_10624kc7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_2_layer_call_fn_10611^c7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_conv2d_3_layer_call_and_return_conditional_losses_10722kf7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_3_layer_call_fn_10709^f7?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_10820ki7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????0
? ?
(__inference_conv2d_4_layer_call_fn_10807^i7?4
-?*
(?%
inputs????????? 
? " ??????????0?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_10928kl7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_5_layer_call_fn_10915^l7?4
-?*
(?%
inputs?????????0
? " ??????????@?
A__inference_conv2d_layer_call_and_return_conditional_losses_10428k]7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????	
? ?
&__inference_conv2d_layer_call_fn_10415^]7?4
-?*
(?%
inputs?????????
? " ??????????	?
@__inference_dense_layer_call_and_return_conditional_losses_10326]450?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? y
%__inference_dense_layer_call_fn_10316P450?-
&?#
!?
inputs??????????
? "??????????
?
B__inference_flatten_layer_call_and_return_conditional_losses_10307a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_10301T7?4
-?*
(?%
inputs?????????@
? "???????????:
__inference_loss_fn_0_10347]?

? 
? "? :
__inference_loss_fn_1_10358`?

? 
? "? :
__inference_loss_fn_2_10369c?

? 
? "? :
__inference_loss_fn_3_10380f?

? 
? "? :
__inference_loss_fn_4_10391i?

? 
? "? :
__inference_loss_fn_5_10402l?

? 
? "? ?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_10902?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_10897?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
#__inference_signature_wrapper_10296? ]^_op`abqrcdestfghuvijkwxlmnyz45C?@
? 
9?6
4
input_1)?&
input_1?????????"3?0
.
output_1"?
output_1?????????
?
B__inference_softmax_layer_call_and_return_conditional_losses_10336\3?0
)?&
 ?
inputs?????????


 
? "%?"
?
0?????????

? z
'__inference_softmax_layer_call_fn_10331O3?0
)?&
 ?
inputs?????????


 
? "??????????
