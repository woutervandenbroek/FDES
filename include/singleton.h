/*==================================================================

Copyright (C) 2015 Wouter Van den Broek, Xiaoming Jiang

This file is part of FDES.

FDES is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FDES is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FDES. If not, see <http://www.gnu.org/licenses/>.

Email: wouter.vandenbroek@uni-ulm.de, wouter.vandenbroek1@gmail.com,
       xiaoming.jiang@uni-ulm.de, jiang.xiaoming1984@gmail.com 

===================================================================*/

#include <cublas_v2.h>
#include "cublas_assert.hpp"

struct cublas_handle_initializer
{
	cublasHandle_t handle;

	cublas_handle_initializer()
	{
		cublas_assert( cublasCreate(&handle) );
	}

	~cublas_handle_initializer()
	{
		cublas_assert( cublasDestroy(handle) );
	}
};

template< typename T >
struct singleton
{
	typedef T value_type;
	typedef singleton self_type;
private:
	singleton( const self_type& );
	self_type& operator = ( const self_type& );
	singleton();

private:
	struct constuctor
	{
		constuctor() { self_type::instance(); }
		inline void null_action() const { }
	};

	static constuctor constuctor_;

public:
	static value_type&
		instance()
	{
		static value_type instance_;
		constuctor_.null_action();
		return instance_;
	}
};

template<typename T>
typename singleton<T>::constuctor singleton<T>::constuctor_;  