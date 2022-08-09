// Copyright (C) 2022 setzer22 and contributors
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::sync::Arc;

use mlua::{AnyUserData, AsChunk, FromLua, Lua, Table, ToLua, UserData};

use crate::{
    graph::NodeDefinitions,
    lua_engine::ToLuaError,
    prelude::{
        halfedge::{
            id_types::{FaceId, HalfEdgeId, VertexId},
            ChannelKeyType, ChannelValueType, HalfEdgeMesh,
        },
        selection::SelectionExpression,
    },
};

/// Convenience macro for registering a lua function inside a global table.
macro_rules! lua_fn {
    ($lua:ident, $table:ident, $name:expr, || -> $retval:ty { $($body:tt)* }) => {
        $table.set($name,
            #[allow(unused_parens)]
            #[allow(unused_variables)]
            $lua.create_function(|$lua, ()| -> mlua::Result<$retval> {
                $($body)*
            })?
        )?
    };
    ($lua:ident, $table:ident, $name:expr, |$($argname:ident : $typ:ty),*| -> $retval:ty { $($body:tt)* }) => {
        $table.set($name,
            #[allow(unused_parens)]
            #[allow(unused_variables)]
            $lua.create_function(|$lua, ($($argname),*) : ($($typ),*)| -> mlua::Result<$retval> {
                $($body)*
            })?
        )?
    };
}

mod runtime_types;
pub use runtime_types::*;

pub mod lua_require_io;
pub use lua_require_io::*;

mod lua_core_library;
mod lua_constructors_library;
mod lua_export_library;
mod lua_mesh_library;
mod lua_primitives_library;

pub mod lua_documentation;

/// A function pointer to register global lua functions. Stored globally using
/// the `inventory` crate.
pub struct LuaRegisterFn {
    pub f: fn(&mlua::Lua) -> mlua::Result<()>,
}
inventory::collect!(LuaRegisterFn);

/// Loads pure Lua libraries that are part of the blackjack core APIs
pub fn load_lua_libraries(lua: &Lua) -> anyhow::Result<()> {
    macro_rules! def_library {
        ($name:expr, $file:expr) => {
            let lib: mlua::Value = lua.load(include_str!($file)).call(())?;
            lua.globals().set($name, lib)?;
        };
    }

    def_library!("NodeLibrary", "node_library.lua");
    Ok(())
}

/// Loads all blackjack Rust function wrappers to the Lua API
pub fn load_host_libraries(lua: &Lua, lua_io: Arc<dyn LuaFileIo + 'static>) -> anyhow::Result<()> {
    lua_core_library::load(lua, lua_io)?;
    lua_mesh_library::load(lua)?;
    lua_primitives_library::load(lua)?;
    lua_export_library::load(lua)?;
    lua_constructors_library::load(lua)?;

    for register_fn in inventory::iter::<LuaRegisterFn>() {
        (register_fn.f)(lua);
    }

    Ok(())
}
